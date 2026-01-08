#!/usr/bin/env python3
"""
测试重新生成功能
"""

import os
import json
from pathlib import Path

def simulate_failed_generation():
    """模拟生成失败的情况，创建一个包含空类别的JSON文件"""
    print("=== 模拟生成失败的场景 ===\n")

    # 创建模拟数据，其中一些类别是空的
    mock_data = {
        "metadata": {
            "description": "Enhanced text prompts for video anomaly detection categories",
            "generator": "CherryStudioPhraseGenerator",
            "total_categories": 13,
            "total_phrases": 35,  # 模拟只有部分成功
            "phrases_per_category": 5,
            "api_provider": "Cherry Studio"
        },
        "categories": {
            "abuse": ["physical violence", "harmful actions", "aggressive behavior"],
            "arrest": [],  # 模拟失败的类别
            "arson": ["fire setting", "property destruction", "flammable liquids"],
            "assault": [],  # 模拟失败的类别
            "burglary": ["breaking entry", "theft from premises"],
            "explosion": [],  # 模拟失败的类别
            "fighting": ["physical confrontation", "mutual combat"],
            "roadAccidents": ["vehicle collision", "traffic incident"],
            "robbery": [],  # 模拟失败的类别
            "shooting": ["firearm discharge", "gunshot sounds"],
            "shoplifting": [],  # 模拟失败的类别
            "stealing": ["unauthorized taking", "property removal"],
            "vandalism": ["property damage", "destruction of objects"]
        }
    }

    # 保存到测试文件
    test_file = "test_partial_generation.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Created test file: {test_file}")

    # 分析失败的类别
    failed_categories = [k for k, v in mock_data["categories"].items() if not v]
    successful_count = len(mock_data["categories"]) - len(failed_categories)
    total_phrases = sum(len(v) for v in mock_data["categories"].values())

    print(f"✓ Total categories: {len(mock_data['categories'])}")
    print(f"✓ Successful categories: {successful_count}")
    print(f"✓ Failed categories: {len(failed_categories)} - {failed_categories}")
    print(f"✓ Total phrases: {total_phrases}")

    return test_file, failed_categories

def test_regenerate_functionality():
    """测试重新生成功能"""
    print("\n=== 测试重新生成功能 ===\n")

    # 注意：这个测试不需要真实的API调用
    # 我们只测试逻辑功能

    try:
        from generate_phrases import CherryStudioPhraseGenerator

        # 使用虚拟API密钥（不会真的调用API）
        gen = CherryStudioPhraseGenerator("dummy-api-key")

        # 测试获取失败类别的功能
        test_file, expected_failed = simulate_failed_generation()

        # 测试获取失败类别
        failed_categories = gen.get_failed_categories(test_file)
        print(f"✓ Detected failed categories: {failed_categories}")

        # 验证检测结果
        if set(failed_categories) == set(expected_failed):
            print("✓ Failed category detection is correct")
        else:
            print("❌ Failed category detection is incorrect")
            print(f"  Expected: {expected_failed}")
            print(f"  Detected: {failed_categories}")

        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✓ Cleaned up test file: {test_file}")

        print("\n✓ Regenerate functionality test completed")

    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_regenerate_functionality()
