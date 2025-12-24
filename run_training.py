#!/usr/bin/env python3
"""
自动执行训练脚本的程序
运行顺序：UCF训练 -> XD训练
包含详细的训练过程记录和日志保存
"""

import subprocess
import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime

def run_command(cmd, description, logger):
    """运行命令并打印状态"""
    logger.info(f"\n{'='*50}")
    logger.info(f"开始执行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    logger.info('='*50)

    start_time = time.time()

    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=False, text=True)
        execution_time = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✓ {description} 执行成功")
            logger.info(f"执行时间: {execution_time:.2f}秒")
            return True
        else:
            logger.error(f"✗ {description} 执行失败，返回码: {result.returncode}")
            return False
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"✗ {description} 执行出错: {str(e)}")
        logger.error(f"执行时间: {execution_time:.2f}秒")
        return False

def main():
    """主函数"""
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)

    # 设置日志记录器
    log_filename = f"logs/training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("VadCLIP 训练会话开始")
    logger.info("执行顺序: UCF数据集训练 -> XD数据集训练")
    logger.info(f"会话日志文件: {log_filename}")
    logger.info("="*60)

    session_start_time = time.time()

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
    logger.info("\n🚀 开始UCF数据集训练阶段")
    ucf_start_time = time.time()
    success1 = run_command(
        [sys.executable, str(ucf_script)],
        "UCF数据集训练",
        logger
    )
    ucf_time = time.time() - ucf_start_time

    if not success1:
        logger.error("\n❌ UCF训练失败，停止执行")
        logger.error(f"UCF训练耗时: {ucf_time:.2f}秒")
        sys.exit(1)
    else:
        logger.info(f"✅ UCF训练成功完成，耗时: {ucf_time:.2f}秒")

    # 步骤2: 运行XD训练
    logger.info("\n🚀 开始XD数据集训练阶段")
    xd_start_time = time.time()
    success2 = run_command(
        [sys.executable, str(xd_script)],
        "XD数据集训练",
        logger
    )
    xd_time = time.time() - xd_start_time

    # 计算总时间
    total_time = time.time() - session_start_time

    if success2:
        logger.info("\n🎉 所有训练任务执行完毕！")
        logger.info("="*60)
        logger.info("训练会话总结:")
        logger.info(f"  UCF训练耗时: {ucf_time:.2f}秒 ({ucf_time/3600:.2f}小时)")
        logger.info(f"  XD训练耗时: {xd_time:.2f}秒 ({xd_time/3600:.2f}小时)")
        logger.info(f"  总训练时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
        logger.info(f"  会话日志已保存到: {log_filename}")
        logger.info("  详细训练日志请查看 logs/ 目录下的对应文件")
        logger.info("="*60)
    else:
        logger.error("\n❌ XD训练失败")
        logger.error(f"XD训练耗时: {xd_time:.2f}秒")
        logger.error(f"会话总耗时: {total_time:.2f}秒")
        sys.exit(1)

if __name__ == "__main__":
    main()
