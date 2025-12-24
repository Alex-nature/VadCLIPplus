#!/usr/bin/env python3
"""
自动执行训练脚本的程序
运行顺序：UCF训练 -> XD训练
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """运行命令并打印状态"""
    print(f"\n{'='*50}")
    print(f"开始执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*50)

    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {description} 执行成功")
            return True
        else:
            print(f"✗ {description} 执行失败，返回码: {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ {description} 执行出错: {str(e)}")
        return False

def main():
    """主函数"""
    print("训练脚本自动执行器")
    print("执行顺序: UCF训练 -> XD训练")

    # 检查脚本文件是否存在
    ucf_script = Path("src/ucf_train.py")
    xd_script = Path("src/xd_train.py")

    if not ucf_script.exists():
        print(f"错误: {ucf_script} 不存在")
        sys.exit(1)

    if not xd_script.exists():
        print(f"错误: {xd_script} 不存在")
        sys.exit(1)

    # 步骤1: 运行UCF训练
    success1 = run_command(
        [sys.executable, str(ucf_script)],
        "UCF数据集训练"
    )

    if not success1:
        print("\nUCF训练失败，停止执行")
        sys.exit(1)

    # 步骤2: 运行XD训练
    success2 = run_command(
        [sys.executable, str(xd_script)],
        "XD数据集训练"
    )

    if success2:
        print("\n🎉 所有训练任务执行完毕！")
    else:
        print("\nXD训练失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
