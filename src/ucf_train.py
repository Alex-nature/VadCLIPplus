import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import logging
import time
import os
import json
from datetime import datetime

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option

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
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    # 创建日志目录
    os.makedirs('model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 设置日志记录器
    log_filename = f"logs/ucf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 初始化训练历史记录
    training_history = {
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'epochs': [],
        'best_ap': 0,
        'best_epoch': 0,
        'total_training_time': 0
    }

    logger.info("="*50)
    logger.info("开始UCF数据集训练")
    logger.info(f"日志文件: {log_filename}")
    logger.info(f"模型保存路径: {args.model_path}")
    logger.info(f"检查点路径: {args.checkpoint_path}")
    logger.info(f"训练参数: lr={args.lr}, batch_size={args.batch_size}, max_epoch={args.max_epoch}")
    logger.info("="*50)

    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0
    total_start_time = time.time()

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        training_history['best_ap'] = ap_best
        training_history['best_epoch'] = epoch
        logger.info("加载检查点信息:")
        logger.info(f"恢复epoch: {epoch+1}, 最佳AP: {ap_best:.4f}")

    for e in range(epoch, args.max_epoch):
        epoch_start_time = time.time()
        logger.info(f"\n开始训练第 {e+1}/{args.max_epoch} 轮")

        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        batch_count = 0

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)

            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1
            loss_total3 += loss3.item()

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_count += 1

            if batch_count % 20 == 0:  # 每20个batch记录一次
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {e+1}, Batch {batch_count}, Loss1: {loss_total1/batch_count:.4f}, Loss2: {loss_total2/batch_count:.4f}, Loss3: {loss_total3/batch_count:.4f}, LR: {current_lr:.6f}")

        # 计算平均loss
        avg_loss1 = loss_total1 / batch_count
        avg_loss2 = loss_total2 / batch_count
        avg_loss3 = loss_total3 / batch_count
        total_loss = avg_loss1 + avg_loss2 + avg_loss3

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 验证
        logger.info(f"Epoch {e+1} - 开始验证...")
        val_start_time = time.time()
        AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        AP = AUC  # 根据代码逻辑，AP = AUC
        val_time = time.time() - val_start_time

        epoch_time = time.time() - epoch_start_time

        # 记录epoch信息
        epoch_info = {
            'epoch': e + 1,
            'train_loss1': avg_loss1,
            'train_loss2': avg_loss2,
            'train_loss3': avg_loss3,
            'total_train_loss': total_loss,
            'auc': AUC,
            'ap': AP,
            'learning_rate': current_lr,
            'epoch_time': epoch_time,
            'val_time': val_time,
            'timestamp': datetime.now().isoformat()
        }
        training_history['epochs'].append(epoch_info)

        logger.info("="*60)
        logger.info(f"Epoch {e+1}/{args.max_epoch} 完成")
        logger.info(f"训练损失 - Loss1: {avg_loss1:.4f}, Loss2: {avg_loss2:.4f}, Loss3: {avg_loss3:.4f}, 总损失: {total_loss:.4f}")
        logger.info(f"验证结果 - AUC: {AUC:.4f}, AP: {AP:.4f}")
        logger.info(f"学习率: {current_lr:.6f}")
        logger.info(f"训练时间: {epoch_time:.2f}s, 验证时间: {val_time:.2f}s")
        logger.info("="*60)

        # 保存最佳模型
        if AP > ap_best:
            ap_best = AP
            training_history['best_ap'] = ap_best
            training_history['best_epoch'] = e + 1

            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)
            logger.info(f"🎉 发现更好的模型! AP: {AP:.4f}, 已保存检查点")

        # 保存当前模型
        torch.save(model.state_dict(), 'model/model_cur.pth')

        # 重新加载最佳模型权重进行下一轮训练
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 保存最终模型
    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

    # 计算总训练时间并保存训练历史
    total_training_time = time.time() - total_start_time
    training_history['total_training_time'] = total_training_time
    training_history['end_time'] = datetime.now().isoformat()

    # 保存训练历史到JSON文件
    history_filename = f"logs/ucf_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_filename, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)

    logger.info("="*50)
    logger.info("UCF训练完成！")
    logger.info(f"总训练时间: {total_training_time:.2f}秒 ({total_training_time/3600:.2f}小时)")
    logger.info(f"最佳AP: {training_history['best_ap']:.4f} (第{training_history['best_epoch']}轮)")
    logger.info(f"最终模型已保存到: {args.model_path}")
    logger.info(f"训练历史已保存到: {history_filename}")
    logger.info("="*50)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)