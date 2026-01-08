#!/usr/bin/env python3
"""
使用CLIP文本编码器计算相似度，对生成的短语进行过滤和排序
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# 导入CLIP相关模块
import sys
sys.path.append('src')
from clip import clip


class PhraseSimilarityFilter:
    """
    使用CLIP计算文本相似度，对短语进行过滤和排序
    """

    def __init__(self, model_name: str = "ViT-B/16", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化相似度过滤器

        Args:
            model_name: CLIP模型名称
            device: 计算设备
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading CLIP model: {model_name} on {device}")
        self.model, self.preprocess = clip.load(model_name, device=device)

        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        print("✓ CLIP model loaded and frozen")

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        使用CLIP编码文本列表

        Args:
            texts: 文本列表

        Returns:
            文本嵌入张量 (N, D)，数据类型为float32
        """
        # 分批处理，避免内存溢出
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # CLIP tokenize
            text_tokens = clip.tokenize(batch_texts).to(self.device)

            # 编码文本
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                # 确保转换为float32类型
                text_features = text_features.float()
                # L2归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(text_features.cpu())

        # 合并所有批次，并确保返回float32类型
        result = torch.cat(all_embeddings, dim=0)
        return result.float()

    def calculate_similarities(self, category_embedding: torch.Tensor,
                             phrase_embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算类别嵌入与短语嵌入之间的相似度

        Args:
            category_embedding: 类别嵌入 (1, D)
            phrase_embeddings: 短语嵌入 (N, D)

        Returns:
            相似度分数 (N,)
        """
        # 确保两个张量都是float32类型
        category_embedding = category_embedding.float()
        phrase_embeddings = phrase_embeddings.float()

        # 计算余弦相似度
        similarities = torch.matmul(phrase_embeddings, category_embedding.T).squeeze(-1)
        return similarities

    def select_topk_phrases(self, category: str, phrases: List[str],
                          top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        为指定类别选择top-k最相似的短语

        Args:
            category: 类别名称
            phrases: 该类别的所有短语
            top_k: 选择的数量

        Returns:
            (选中的短语列表, 对应的相似度分数列表)
        """
        if not phrases:
            return [], []

        # 编码类别和短语
        all_texts = [category] + phrases
        embeddings = self.encode_texts(all_texts)

        # 分离类别和短语嵌入
        category_embedding = embeddings[0:1]  # (1, D)
        phrase_embeddings = embeddings[1:]    # (N, D)

        # 确保两个张量都在同一设备上且类型相同
        if category_embedding.device != phrase_embeddings.device:
            category_embedding = category_embedding.to(phrase_embeddings.device)

        # 计算相似度
        similarities = self.calculate_similarities(category_embedding, phrase_embeddings)

        # 选择top-k
        top_k = min(top_k, len(similarities))
        top_values, top_indices = torch.topk(similarities, top_k)

        # 获取对应的短语和相似度
        selected_phrases = [phrases[idx] for idx in top_indices.tolist()]
        selected_similarities = top_values.tolist()

        return selected_phrases, selected_similarities

    def filter_all_categories(self, input_file: str = "enhanced_prompts.json",
                            output_file: str = "filtered_prompts.json",
                            top_k: int = 5) -> Dict[str, Dict]:
        """
        过滤所有类别的短语，选择top-k最相似的

        Args:
            input_file: 输入JSON文件路径
            output_file: 输出JSON文件路径
            top_k: 每个类别保留的短语数量

        Returns:
            过滤后的结果字典
        """
        print(f"Loading phrases from {input_file}...")

        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        categories = data.get('categories', {})
        print(f"Found {len(categories)} categories to process")

        filtered_results = {}
        total_phrases_selected = 0

        # 处理每个类别
        for category, phrases in categories.items():
            print(f"\nProcessing category: '{category}' ({len(phrases)} phrases)")

            # 选择top-k短语
            selected_phrases, similarities = self.select_topk_phrases(category, phrases, top_k)

            # 保存结果
            filtered_results[category] = {
                'phrases': selected_phrases,
                'similarities': similarities,
                'original_count': len(phrases),
                'selected_count': len(selected_phrases)
            }

            total_phrases_selected += len(selected_phrases)

            # 显示结果
            print(f"  ✓ Selected {len(selected_phrases)} top phrases:")
            for i, (phrase, sim) in enumerate(zip(selected_phrases, similarities), 1):
                print(f"    {i}. \"{phrase}\"")
                print(f"       Similarity: {sim:.4f}")

            # 显示相似度统计
            sim_array = np.array(similarities)
            print(f"       Similarity stats - Min: {sim_array.min():.4f}, Max: {sim_array.max():.4f}, Avg: {sim_array.mean():.4f}")

        # 构建输出数据结构
        output_data = {
            "metadata": {
                "description": "Filtered text prompts based on CLIP similarity",
                "source_file": input_file,
                "clip_model": self.model_name,
                "total_categories": len(filtered_results),
                "total_phrases_selected": total_phrases_selected,
                "top_k_per_category": top_k,
                "filter_method": "cosine_similarity"
            },
            "categories": filtered_results
        }

        # 保存结果
        print(f"\nSaving filtered results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print("✓ Filtering completed!")
        print(f"  - Total categories: {len(filtered_results)}")
        print(f"  - Total phrases selected: {total_phrases_selected}")

        return output_data


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="Filter phrases by CLIP similarity")
    parser.add_argument("--input", type=str, default="enhanced_prompts.json",
                       help="Input JSON file with phrases")
    parser.add_argument("--output", type=str, default="filtered_prompts.json",
                       help="Output JSON file for filtered results")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top phrases to select per category")
    parser.add_argument("--model", type=str, default="ViT-B/16",
                       help="CLIP model to use")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (auto-detect if not specified)")

    args = parser.parse_args()

    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== CLIP Phrase Similarity Filter ===")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Top-k per category: {args.top_k}")
    print(f"CLIP model: {args.model}")
    print(f"Device: {args.device}")
    print()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    # 初始化过滤器
    try:
        filter = PhraseSimilarityFilter(model_name=args.model, device=args.device)
    except Exception as e:
        print(f"❌ Failed to initialize CLIP model: {e}")
        return

    # 执行过滤
    try:
        results = filter.filter_all_categories(
            input_file=args.input,
            output_file=args.output,
            top_k=args.top_k
        )

        print("\n🎉 Filtering completed successfully!")
        print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Filtering failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()