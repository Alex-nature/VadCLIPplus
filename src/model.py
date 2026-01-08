from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj

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
        # fusion MLP to combine visual and t_guided (concat -> project back to visual_width)
        self.fusion_mlp = nn.Sequential(OrderedDict([
            ("f_fc1", nn.Linear(visual_width + self.embed_dim, visual_width * 2)),
            ("f_gelu", QuickGELU()),
            ("f_fc2", nn.Linear(visual_width * 2, visual_width))
        ]))

        self.initialize_parameters()
        # per-class learnable alpha (unconstrained, apply sigmoid to map to (0,1))
        self.alpha_param = nn.Parameter(torch.full((self.num_class,), 0.7))

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

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def encode_category_embeddings(self, text):
        """
        生成类别的CLIP embeddings，并使用过滤后的短语进行增强

        Args:
            text: 类别名称列表，如['normal', 'abuse', 'arrest', ...]

        Returns:
            类别embeddings (num_classes, embed_dim)
        """
        # 首先生成基础的类别embeddings
        category_tokens = clip.tokenize(text).to(self.device)
        word_embeddings = self.clipmodel.encode_token(category_tokens)

        # 创建简单的text_embeddings：只包含[CLS] + 词嵌入
        batch_size = len(text)
        text_embeddings = torch.zeros(batch_size, 77, self.embed_dim).to(self.device)
        text_tokens = torch.zeros(batch_size, 77).to(self.device).long()

        for i in range(batch_size):
            # [CLS] token embedding
            text_embeddings[i, 0] = word_embeddings[i, 0]  # BOS token

            # 找到实际文本的结束位置
            eos_pos = (category_tokens[i] == 49407).nonzero(as_tuple=True)[0]  # EOS token
            if len(eos_pos) > 0:
                text_len = eos_pos[0].item()
            else:
                text_len = min(75, (category_tokens[i] != 0).sum().item())

            # 复制词嵌入到对应位置
            text_embeddings[i, 1:text_len] = word_embeddings[i, 1:text_len]
            text_tokens[i, :text_len] = category_tokens[i, :text_len]

            # EOS token
            text_embeddings[i, text_len] = word_embeddings[i, text_len]
            text_tokens[i, text_len] = category_tokens[i, text_len]

        with torch.no_grad():
            category_embeddings = self.clipmodel.encode_text(text_embeddings, text_tokens)
            category_embeddings = category_embeddings / category_embeddings.norm(dim=-1, keepdim=True)

        # 使用过滤后的短语增强异常类别的embeddings
        enhanced_embeddings = self.enhance_embeddings_with_phrases(category_embeddings, text)

        return enhanced_embeddings

    def enhance_embeddings_with_phrases(self, category_embeddings, text):
        """
        使用过滤后的短语增强类别embeddings

        Args:
            category_embeddings: 基础类别embeddings (num_classes, embed_dim)
            text: 类别名称列表

        Returns:
            增强后的embeddings
        """
        enhanced_embeddings = category_embeddings.clone()

        # 尝试加载过滤后的短语数据
        try:
            import json
            import os
            json_path = os.path.join(os.path.dirname(__file__), '..', 'filtered_prompts.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    filtered_data = json.load(f)

                categories_data = filtered_data.get('categories', {})

                # 为每个类别进行增强（跳过normal类）
                for i, category_name in enumerate(text):
                    if category_name.lower() == 'normal':
                        continue  # 不增强normal类

                    if category_name in categories_data:
                        # 获取该类别的过滤后短语
                        phrases = categories_data[category_name]['phrases']
                        if phrases:
                            # 为短语生成embeddings
                            phrase_tokens = clip.tokenize(phrases).to(self.device)
                            with torch.no_grad():
                                # 为短语创建简单的text_embeddings
                                num_phrases = len(phrases)
                                phrase_text_embeddings = torch.zeros(num_phrases, 77, self.embed_dim).to(self.device)
                                phrase_text_tokens = torch.zeros(num_phrases, 77).to(self.device).long()

                                # 获取短语的词嵌入
                                phrase_word_embeddings = self.clipmodel.encode_token(phrase_tokens)

                                for j in range(num_phrases):
                                    # [CLS] token
                                    phrase_text_embeddings[j, 0] = phrase_word_embeddings[j, 0]

                                    # 找到短语结束位置
                                    eos_pos = (phrase_tokens[j] == 49407).nonzero(as_tuple=True)[0]
                                    if len(eos_pos) > 0:
                                        phrase_len = eos_pos[0].item()
                                    else:
                                        phrase_len = min(75, (phrase_tokens[j] != 0).sum().item())

                                    # 复制词嵌入
                                    phrase_text_embeddings[j, 1:phrase_len] = phrase_word_embeddings[j, 1:phrase_len]
                                    phrase_text_tokens[j, :phrase_len] = phrase_tokens[j, :phrase_len]

                                    # EOS token
                                    phrase_text_embeddings[j, phrase_len] = phrase_word_embeddings[j, phrase_len]
                                    phrase_text_tokens[j, phrase_len] = phrase_tokens[j, phrase_len]

                                # 编码短语
                                phrase_embeddings = self.clipmodel.encode_text(phrase_text_embeddings, phrase_text_tokens)
                                phrase_embeddings = phrase_embeddings / phrase_embeddings.norm(dim=-1, keepdim=True)

                            # 使用 phrase_embeddings 与 category_embedding 的余弦相似度作为权重
                            # phrase_embeddings: (num_phrases, D)
                            # category_embeddings[i]: (D,)
                            # 注意 dtype/device 对齐
                            cat_emb = category_embeddings[i].to(phrase_embeddings.dtype)
                            # similarities by cosine (since both are normalized)
                            sims = torch.matmul(phrase_embeddings, cat_emb)  # (num_phrases,)
                            weights = torch.softmax(sims, dim=0)  # (num_phrases,)

                            # 加权组合短语embeddings
                            weighted_phrase_emb = torch.sum(weights.unsqueeze(-1) * phrase_embeddings, dim=0)

                            # 使用可学习的 alpha（sigmoid 参数化保证在 (0,1)）
                            alpha_raw = self.alpha_param[i] if hasattr(self, 'alpha_param') else torch.tensor(0.7, device=weighted_phrase_emb.device)
                            alpha = torch.sigmoid(alpha_raw).to(weighted_phrase_emb.dtype)

                            # 融合并归一化
                            enhanced = alpha * category_embeddings[i] + (1.0 - alpha) * weighted_phrase_emb
                            enhanced_embeddings[i] = enhanced / (enhanced.norm(dim=-1, keepdim=True) + 1e-12)

        except Exception as e:
            print(f"Warning: Failed to load enhanced phrases: {e}")
            # 如果加载失败，返回原始的类别embeddings
            pass

        return enhanced_embeddings

    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)

        text_features_ori = self.encode_textprompt(text)
        # 使用text_feature作为后续操作的变量
        text_features = text_features_ori

        # Step 1: compute t_guided using class-text -> visual conditional weighting
        # visual_features: (B, T, V)
        # text_features: (C, D)
        # produce w: (B, T, C) and t_guided: (B, T, D)
        tau = 0.07
        # L2 normalize visual and text features (per-vector)
        visual_norm = visual_features / (visual_features.norm(dim=-1, keepdim=True) + 1e-12)
        text_feat = text_features  # (C, D)
        text_norm = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        # ensure dtype/device alignment
        text_norm = text_norm.to(visual_norm.dtype)

        # logits: (B, T, C)
        logits_w = torch.matmul(visual_norm, text_norm.T) / tau
        w = torch.softmax(logits_w, dim=-1)  # (B, T, C)

        # weighted sum to produce text guidance per timestep: (B, T, D)
        t_guided = torch.matmul(w, text_feat.to(visual_norm.dtype))

        # Fuse visual and t_guided via concat + fusion_mlp
        # visual_features: (B, T, V), t_guided: (B, T, D) where D==V
        multimodal = torch.cat([visual_features, t_guided], dim=-1)  # (B, T, V+D)
        fused = self.fusion_mlp(multimodal)  # (B, T, V)

        logits1 = self.classifier(fused + self.mlp2(fused))

        # text参数就是类别名称列表，如['normal', 'abuse', 'arrest', ...]
        category_embeddings = self.encode_category_embeddings(text)

        # use fused multimodal features for alignment with category embeddings
        fused_norm = fused / (fused.norm(dim=-1, keepdim=True) + 1e-12)
        logits2 = fused_norm @ category_embeddings.type(fused_norm.dtype).T / 0.07

        return text_features_ori, logits1, logits2
    