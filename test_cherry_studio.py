#!/usr/bin/env python3
"""
测试Cherry Studio连接和基本功能
"""

import os
import sys
from pathlib import Path

def test_env_config():
    """测试环境配置"""
    print("=== 环境配置测试 ===\n")

    # 检查.env文件
    env_file = Path(".env")
    if env_file.exists():
        print("✓ 找到 .env 文件")
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'CHERRY_STUDIO_API_KEY=' in content:
                    # 检查是否设置了实际的密钥（不是示例）
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith('CHERRY_STUDIO_API_KEY='):
                            key_value = line.split('=', 1)[1].strip()
                            if key_value and key_value != 'your-cherry-studio-api-key-here':
                                print("✓ API密钥已配置")
                                return True
                            else:
                                print("❌ API密钥未设置（仍为示例值）")
                                return False
                else:
                    print("❌ .env文件中未找到CHERRY_STUDIO_API_KEY")
                    return False
        except Exception as e:
            print(f"❌ 读取.env文件失败: {e}")
            return False
    else:
        print("❌ 未找到 .env 文件")
        print("请复制 env_cherry_studio_example.txt 为 .env 并配置API密钥")
        return False

def test_import():
    """测试模块导入"""
    print("\n=== 模块导入测试 ===\n")

    try:
        from generate_phrases import CherryStudioPhraseGenerator
        print("✓ CherryStudioPhraseGenerator 导入成功")

        # 测试初始化（不调用API）
        if test_env_config():
            api_key = None
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('CHERRY_STUDIO_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break

            if api_key:
                try:
                    generator = CherryStudioPhraseGenerator(api_key)
                    print("✓ 生成器初始化成功")
                    return True
                except Exception as e:
                    print(f"❌ 生成器初始化失败: {e}")
                    return False
            else:
                print("❌ 未找到API密钥")
                return False
        else:
            print("❌ 环境配置测试失败，跳过初始化测试")
            return False

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== Cherry Studio LLM短语生成器测试 ===\n")

    env_ok = test_env_config()
    import_ok = test_import()

    print("\n=== 测试结果汇总 ===")
    print(f"环境配置: {'✓' if env_ok else '❌'}")
    print(f"模块导入: {'✓' if import_ok else '❌'}")

    if env_ok and import_ok:
        print("\n🎉 所有测试通过！可以开始生成短语了。")
        print("运行命令: python generate_phrases.py")
    else:
        print("\n❌ 测试失败，请检查配置后再试。")
        print("参考: README_CHERRY_STUDIO.md")

if __name__ == "__main__":
    main()
