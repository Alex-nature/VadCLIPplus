from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj

from utils.tools import get_batch_mask
from utils.adapter_modules import SimpleAdapter, SimpleProj
from utils.descriptions import DESCRIPTIONS_ORI, DESCRIPTIONS_ORI_XD

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class CLIP_Adapter(nn.Module):
    def __init__(self, clipmodel, device, text_adapt_until=3, t_w=0.1):
        super(CLIP_Adapter, self).__init__()
        self.clipmodel = clipmodel
        self.text_adapt_until = text_adapt_until
        self.t_w = t_w
        self.device = device

        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(512, 512) for _ in range(text_adapt_until)] +
            [SimpleProj(512, 512, relu=True)]
        )

        self._init_weights_()

    def _init_weights_(self):
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)

        cast_dtype = self.clipmodel.token_embedding.weight.dtype

        x = self.clipmodel.token_embedding(text).to(cast_dtype) 

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2) 

        for i in range(len(self.clipmodel.transformer.resblocks)):
            x = self.clipmodel.transformer.resblocks[i](x)
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) /
                    (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x

        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        eot_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices]
        x = self.text_adapter[-1](x)

        return x


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 args,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.clip_adapter = CLIP_Adapter(self.clipmodel, self.device, args.text_adapt_until, args.t_w)
        self._text_features_cache = None

        self.tv_attn = nn.MultiheadAttention(embed_dim=visual_width, num_heads=visual_head, batch_first=True)
        self.tv_ln_t = LayerNorm(visual_width)
        self.tv_ln_v = LayerNorm(visual_width)
        self.alpha_tv = nn.Parameter(torch.tensor(0.1))  # 融合强度，可学/可固定
        self.tv_mlp = nn.Sequential(
            nn.Linear(visual_width, visual_width * 4),
            QuickGELU(),
            nn.Linear(visual_width * 4, visual_width),
        )



        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def get_text_features(self, text):
        if not self.training and self._text_features_cache is not None:
            return self._text_features_cache

        category_features = []
        if len(text) == 14:
            DESCRIPTIONS = DESCRIPTIONS_ORI
        else:
            DESCRIPTIONS = DESCRIPTIONS_ORI_XD
        for class_name, descriptions in DESCRIPTIONS.items():
            tokens = clip.tokenize(descriptions).to(self.device)

            text_features = self.clip_adapter.encode_text(tokens)
            mean_feature = text_features.mean(dim=0)
            mean_feature = mean_feature / mean_feature.norm()
            category_features.append(mean_feature)
        text_features_ori = torch.stack(category_features, dim=0)

        if not self.training:
            self._text_features_cache = text_features_ori
            
        return text_features_ori


    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)          # [B,T,D]
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))     # [B,T,1]

        B, T, D = visual_features.shape
        lengths_t = torch.as_tensor(lengths, device=visual_features.device, dtype=torch.long)
        key_padding_mask = get_batch_mask(lengths_t, T).to(visual_features.device)  # [B,T] True=pad

        text_features_ori = self.get_text_features(text)                            # [C,D]
        text_tokens = text_features_ori.unsqueeze(0).expand(B, -1, -1)              # [B,C,D]

        # =========================================================
        # 策略1：正常原型消除 + 残差放大（对 visual_features 做增强）
        # =========================================================
        tau_w = 1.0
        gamma = 0.5

        # w: 异常时间分布（mask + detach）
        w = logits1.squeeze(-1).detach()                                            # [B,T]
        w = w.masked_fill(key_padding_mask, float('-inf'))                          # pad -> -inf
        w = torch.softmax(w / tau_w, dim=1)                                         # [B,T]

        # wn: 正常时间分布（mask掉padding并归一化）
        wn = (1.0 - w)                                                              # [B,T]
        wn = wn.masked_fill(key_padding_mask, 0.0)                                  # pad -> 0
        wn = wn / (wn.sum(dim=1, keepdim=True) + 1e-6)                              # [B,T]

        # p_no: 正常原型
        p_no = torch.einsum('bt,btd->bd', wn, visual_features)                       # [B,D]

        # visual_features_enh: 残差增强
        visual_features_enh = visual_features + gamma * (visual_features - p_no.unsqueeze(1))  # [B,T,D]

        # =========================================================
        # Cross-Attention（用增强后的视觉特征 + key_padding_mask）
        # =========================================================
        V = self.tv_ln_v(visual_features_enh)                                        # [B,T,D]
        Tt = self.tv_ln_t(text_tokens)                                               # [B,C,D]

        ctx, _ = self.tv_attn(
            query=Tt, key=V, value=V,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )                                                                            # [B,C,D]

        # （可选）你原来的 gating：只放大，不改变关注时序
        g = (w.max(dim=1).values).unsqueeze(-1).unsqueeze(-1)                        # [B,1,1]
        ctx = ctx * (1.0 + g)

        # 文本融合
        text_features = text_tokens + self.alpha_tv * ctx
        text_features = text_features + self.tv_mlp(text_features)                   # [B,C,D]

        # logits2：用增强后的视觉特征（更分离）
        visual_features_norm = visual_features_enh / (visual_features_enh.norm(dim=-1, keepdim=True) + 1e-6)
        text_features_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        logits2 = visual_features_norm @ text_features_norm.permute(0, 2, 1).type(visual_features_norm.dtype) / 0.07  # [B,T,C]

        # =========================================================
        # 用字典返回：用于训练脚本计算 L_sepV 的最小集合
        # =========================================================
        aux = {
            "visual_features": visual_features,           # [B,T,D]  (策略2推荐用增强前)
            "w": w,                                       # [B,T]
            "wn": wn,                                     # [B,T]
            "p_no": p_no,                                 # [B,D]   (可选：减少训练端重复算)
            "key_padding_mask": key_padding_mask,         # [B,T]   (可选：后续若要mask logits2)
            "w_max": w.max(dim=1).values,                 # [B]     (可选：gate L_sepV)
            # 如果你想在训练端尝试用增强后特征算分离，也把它返回：
            # "visual_features_enh": visual_features_enh,  # [B,T,D]
        }

        return text_features_ori, logits1, logits2, aux

