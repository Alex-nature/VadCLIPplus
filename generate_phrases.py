#!/usr/bin/env python3
"""
使用Cherry Studio API为异常检测类别生成LLM增强的短语
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, List

class CherryStudioPhraseGenerator:
    """
    使用Cherry Studio API生成异常检测类别的描述短语
    """

    def __init__(self, api_key: str, base_url: str = "https://chat.cloudapi.vip"):
        """
        初始化Cherry Studio生成器

        Args:
            api_key: Cherry Studio API密钥
            base_url: Cherry Studio服务器地址
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_phrase_for_category(self, category: str, num_phrases: int = 5) -> List[str]:
        """
        为单个类别生成描述短语

        Args:
            category: 异常类别名称
            num_phrases: 生成短语数量

        Returns:
            短语列表
        """
        prompt = f"""
        Generate {num_phrases} different descriptive phrases for the anomaly category "{category}" in video anomaly detection.

        Requirements:
        1. Each phrase should describe specific manifestations of this abnormal behavior
        2. Phrases should be concise and clear, between 3-8 words in length
        3. Avoid using the category name itself
        4. Cover different aspects of behavior (actions, context, consequences, etc.)
        5. Ensure diversity and accuracy of phrases

        Return exactly {num_phrases} phrases, one per line, without additional explanations.
        """

        payload = {
            "model": "gpt-5.2",  # 或其他可用的模型
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional expert in video anomaly detection, capable of accurately describing various abnormal behaviors."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": False
        }

        try:
            url = f"{self.base_url}/v1/chat/completions"
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"].strip()
                phrases = [line.strip() for line in response_text.split('\n') if line.strip()]
                return phrases[:num_phrases]  # 确保返回正确数量
            else:
                print(f"Warning: Unexpected API response for category '{category}'")
                return []

        except requests.exceptions.RequestException as e:
            print(f"API request failed for category '{category}': {e}")
            return []
        except Exception as e:
            print(f"Error processing category '{category}': {e}")
            return []

    def generate_all_categories(self, num_phrases: int = 5) -> Dict[str, List[str]]:
        """
        为所有UCF-Crime异常类别生成短语

        Args:
            num_phrases: 每个类别生成的短语数量

        Returns:
            类别到短语列表的字典
        """
        # UCF-Crime数据集的异常类别
        categories = [
            'abuse', 'arrest', 'arson', 'assault', 'burglary', 'explosion',
            'fighting', 'roadAccidents', 'robbery', 'shooting', 'shoplifting',
            'stealing', 'vandalism'
        ]

        results = {}

        print(f"Starting phrase generation for {len(categories)} categories...")
        print(f"Generating {num_phrases} phrases per category\n")

        for i, category in enumerate(categories, 1):
            print(f"[{i}/{len(categories)}] Generating phrases for '{category}'...")

            phrases = self.generate_phrase_for_category(category, num_phrases)

            if phrases:
                results[category] = phrases
                print(f"  ✓ Generated {len(phrases)} phrases")
                for j, phrase in enumerate(phrases, 1):
                    print(f"    {j}. {phrase}")
            else:
                print(f"  ✗ Failed to generate phrases for '{category}'")
                results[category] = []  # 空列表表示失败

            # 避免API限制
            if i < len(categories):  # 最后一个不需要等待
                time.sleep(1)

        return results

    def save_to_json(self, phrases_dict: Dict[str, List[str]], output_file: str = "enhanced_prompts.json"):
        """
        保存生成的短语到JSON文件

        Args:
            phrases_dict: 类别到短语的字典
            output_file: 输出文件名
        """
        # 构建完整的JSON结构
        data = {
            "metadata": {
                "description": "Enhanced text prompts for video anomaly detection categories",
                "generator": "CherryStudioPhraseGenerator",
                "total_categories": len(phrases_dict),
                "total_phrases": sum(len(phrases) for phrases in phrases_dict.values()),
                "phrases_per_category": len(list(phrases_dict.values())[0]) if phrases_dict else 0,
                "api_provider": "Cherry Studio"
            },
            "categories": phrases_dict
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"\n✓ Phrases saved to {output_file}")
            print(f"  - Categories: {len(phrases_dict)}")
            print(f"  - Total phrases: {sum(len(phrases) for phrases in phrases_dict.values())}")

        except Exception as e:
            print(f"✗ Failed to save file: {e}")

    def load_from_json(self, json_file: str) -> Dict[str, List[str]]:
        """
        从JSON文件加载短语

        Args:
            json_file: JSON文件名

        Returns:
            类别到短语的字典
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "categories" in data:
                phrases_dict = data["categories"]
                print(f"✓ Loaded {len(phrases_dict)} categories from {json_file}")

                # 显示统计信息
                if "metadata" in data:
                    meta = data["metadata"]
                    print(f"  - Total phrases: {meta.get('total_phrases', 0)}")
                    print(f"  - Generator: {meta.get('generator', 'Unknown')}")

                return phrases_dict
            else:
                print(f"✗ Invalid JSON format in {json_file}")
                return {}

        except Exception as e:
            print(f"✗ Failed to load JSON file: {e}")
            return {}

    def get_failed_categories(self, json_file: str = "enhanced_prompts.json") -> List[str]:
        """
        获取生成失败的类别（短语列表为空的类别）

        Args:
            json_file: JSON文件路径

        Returns:
            失败类别的列表
        """
        phrases_dict = self.load_from_json(json_file)
        failed_categories = [category for category, phrases in phrases_dict.items() if not phrases]
        return failed_categories

    def regenerate_failed_categories(self, json_file: str = "enhanced_prompts.json", num_phrases: int = 5) -> Dict[str, List[str]]:
        """
        重新生成失败的类别短语

        Args:
            json_file: 现有的JSON文件路径
            num_phrases: 每个类别生成的短语数量

        Returns:
            更新后的短语字典
        """
        # 获取失败的类别
        failed_categories = self.get_failed_categories(json_file)

        if not failed_categories:
            print("✓ All categories have been successfully generated!")
            return self.load_from_json(json_file)

        print(f"Found {len(failed_categories)} failed categories: {failed_categories}")
        print(f"Regenerating {num_phrases} phrases for each failed category...\n")

        # 加载现有数据
        phrases_dict = self.load_from_json(json_file)

        # 重新生成失败的类别
        updated_count = 0
        for i, category in enumerate(failed_categories, 1):
            print(f"[{i}/{len(failed_categories)}] Regenerating phrases for '{category}'...")

            phrases = self.generate_phrase_for_category(category, num_phrases)

            if phrases:
                phrases_dict[category] = phrases
                updated_count += 1
                print(f"  ✓ Successfully regenerated {len(phrases)} phrases")
                for j, phrase in enumerate(phrases, 1):
                    print(f"    {j}. {phrase}")
            else:
                print(f"  ✗ Failed to regenerate phrases for '{category}'")

            # 避免API限制
            if i < len(failed_categories):
                time.sleep(1)

        # 保存更新后的数据
        if updated_count > 0:
            self.save_to_json(phrases_dict, json_file)
            print(f"\n✓ Updated {updated_count} categories in {json_file}")

        return phrases_dict


def main():
    """
    主函数
    """
    import argparse

    parser = argparse.ArgumentParser(description="Cherry Studio LLM Phrase Generator")
    parser.add_argument("--regenerate-failed", action="store_true",
                       help="Only regenerate phrases for failed categories")
    parser.add_argument("--num-phrases", type=int, default=20,
                       help="Number of phrases per category (default: 5)")
    parser.add_argument("--output", type=str, default="enhanced_prompts.json",
                       help="Output JSON file path")

    args = parser.parse_args()

    print("=== Cherry Studio LLM Phrase Generator ===\n")

    # 获取API密钥
    api_key = os.getenv("CHERRY_STUDIO_API_KEY")
    if not api_key:
        # 尝试从.env文件读取
        env_file = Path(".env")
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and 'CHERRY_STUDIO_API_KEY=' in line:
                            api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                            break
            except Exception as e:
                print(f"Error reading .env file: {e}")

    if not api_key:
        print("❌ API key not found. Please set CHERRY_STUDIO_API_KEY environment variable")
        print("or add it to .env file:")
        print("CHERRY_STUDIO_API_KEY=your-api-key-here")
        return

    # 初始化生成器
    generator = CherryStudioPhraseGenerator(api_key)

    if args.regenerate_failed:
        # 只重新生成失败的类别
        print("🔄 Regenerating failed categories...\n")
        phrases_dict = generator.regenerate_failed_categories(args.output, args.num_phrases)

        # 显示结果
        failed_categories = generator.get_failed_categories(args.output)
        successful_categories = len(phrases_dict) - len(failed_categories)
        total_phrases = sum(len(phrases) for phrases in phrases_dict.values())

        print("\n=== Regeneration Summary ===")
        print(f"✓ Successfully processed {len(phrases_dict)} categories")
        print(f"✓ Remaining failed categories: {len(failed_categories)}")
        if failed_categories:
            print(f"  Failed: {failed_categories}")
        print(f"✓ Total phrases: {total_phrases}")

    else:
        # 完整生成所有类别的短语
        phrases_dict = generator.generate_all_categories(num_phrases=args.num_phrases)

        if phrases_dict:
            # 保存结果
            generator.save_to_json(phrases_dict, args.output)

            # 显示摘要
            print("\n=== Generation Summary ===")
            successful_categories = sum(1 for phrases in phrases_dict.values() if phrases)
            total_phrases = sum(len(phrases) for phrases in phrases_dict.values())

            print(f"✓ Successfully generated phrases for {successful_categories}/{len(phrases_dict)} categories")
            print(f"✓ Total phrases generated: {total_phrases}")
            print(f"✓ Results saved to {args.output}")
        else:
            print("❌ No phrases were generated")


if __name__ == "__main__":
    main()