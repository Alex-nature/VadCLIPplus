import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])  # anomaly=1 normal=0
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


# -------- 向量化 方案1（lengths-only, per-sample k）--------
def L_sepV_framewise_vec(
    aux, labels, lengths,
    m_ab=0.2, lam=0.5,
    k_no_min=8, k_no_mul=3,
    K_AB_MAX=16, K_NO_MAX=64
):
    V = aux["visual_features"]
    w = aux["w"]
    device = V.device
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)

    is_anom = (1.0 - labels[:, 0]).float()
    idx = (is_anom > 0.5).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return V.new_tensor(0.0)

    V = V[idx]
    w = w[idx]
    L = lengths[idx]
    Ba, T, D = V.shape

    t_idx = torch.arange(T, device=device).unsqueeze(0)
    valid = (t_idx < L.unsqueeze(1))

    k_ab_i = (L // 16 + 1).clamp(min=1)
    k_ab_i = torch.minimum(k_ab_i, torch.full_like(k_ab_i, K_AB_MAX))

    k_no_i = torch.maximum(torch.full_like(k_ab_i, k_no_min), k_ab_i * k_no_mul)
    k_no_i = torch.minimum(k_no_i, torch.full_like(k_no_i, K_NO_MAX))

    w_top = w.masked_fill(~valid, float('-inf'))
    K_ab = min(K_AB_MAX, T)
    _, top_idx = torch.topk(w_top, k=K_ab, dim=1, largest=True)
    V_ab = V.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))
    r_ab = torch.arange(K_ab, device=device).unsqueeze(0)
    mask_ab = (r_ab < k_ab_i.unsqueeze(1)).float()

    w_bot = w.masked_fill(~valid, float('inf'))
    K_no = min(K_NO_MAX, T)
    _, bot_idx = torch.topk(w_bot, k=K_no, dim=1, largest=False)
    V_no = V.gather(1, bot_idx.unsqueeze(-1).expand(-1, -1, D))
    r_no = torch.arange(K_no, device=device).unsqueeze(0)
    mask_no = (r_no < k_no_i.unsqueeze(1)).float()

    cnt_no = mask_no.sum(dim=1).clamp(min=1.0)
    p_no = (V_no * mask_no.unsqueeze(-1)).sum(dim=1) / cnt_no.unsqueeze(-1)
    p_no = p_no / (p_no.norm(dim=-1, keepdim=True) + 1e-6)

    V_ab_n = V_ab / (V_ab.norm(dim=-1, keepdim=True) + 1e-6)
    V_no_n = V_no / (V_no.norm(dim=-1, keepdim=True) + 1e-6)

    cos_ab = (V_ab_n * p_no.unsqueeze(1)).sum(dim=-1)
    cos_no = (V_no_n * p_no.unsqueeze(1)).sum(dim=-1)

    cnt_ab = mask_ab.sum(dim=1).clamp(min=1.0)
    loss_ab_i = (F.relu(m_ab + cos_ab) * mask_ab).sum(dim=1) / cnt_ab
    loss_no_i = ((1.0 - cos_no) * mask_no).sum(dim=1) / cnt_no

    return (loss_ab_i + lam * loss_no_i).mean()


def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    ap_best = 0
    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", checkpoint['epoch'] + 1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1, loss_total2, loss_total_sepV = 0.0, 0.0, 0.0

        for i, item in enumerate(train_loader):
            visual_feat, text_labels_raw, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels_raw, prompt_text, label_map).to(device)

            text_features, logits1, logits2, aux = model(visual_feat, None, prompt_text, feat_lengths)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6

            eta = 0.1
            loss_sepV = L_sepV_framewise_vec(
                aux, text_labels, feat_lengths,
                m_ab=0.2, lam=0.5,
                k_no_min=8, k_no_mul=3,
                K_AB_MAX=16, K_NO_MAX=64
            )
            loss_total_sepV += loss_sepV.item()

            loss = loss1 + loss2 + loss3 * 1e-4 + eta * loss_sepV

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = (i + 1) * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                print(
                    'epoch: ', e + 1,
                    '| step: ', step,
                    '| loss1: ', loss_total1 / (i + 1),
                    '| loss2: ', loss_total2 / (i + 1),
                    '| loss3: ', loss3.item(),
                    '| sepV: ', loss_total_sepV / (i + 1),
                    '| eta*sepV: ', eta * (loss_total_sepV / (i + 1))
                )

        scheduler.step()
        AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

        if AP > ap_best:
            ap_best = AP
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best
            }
            torch.save(checkpoint, args.checkpoint_path)

        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, args, device
    )

    train(model, train_loader, test_loader, args, label_map, device)
