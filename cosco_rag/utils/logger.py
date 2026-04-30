# utils/logger.py
import sys
import os
from loguru import logger
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 移除默认的控制台输出
logger.remove()

# 控制台输出（开发环境可带颜色）
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True
)

# 全量日志文件（JSON格式，更结构化）
logger.add(
    os.path.join(LOG_DIR, "app_{time:YYYY-MM-DD}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    rotation="00:00",     # 每天午夜轮转
    retention="30 days",  # 保留30天
    compression="gz",     # 压缩旧日志
    level="INFO"
)

# 错误日志单独文件
logger.add(
    os.path.join(LOG_DIR, "error_{time:YYYY-MM-DD}.log"),
    level="ERROR",
    rotation="1 day",
    retention="90 days",
    backtrace=True,       # 异常时记录完整堆栈
    diagnose=True
)

# 可选：JSON 格式额外输出（用于 ELK）
# logger.add(
#     os.path.join(LOG_DIR, "structured.log"),
#     format="{time} | {level} | {extra[thread_id]} | {message}",
#     filter=lambda record: "json" in record["extra"],
#     serialize=True        # 输出 JSON 行
# )

# 全局获取 logger 函数
def get_logger(name: str = None):
    """返回 loguru 的 logger 对象，可添加 name 绑定"""
    if name:
        return logger.bind(name=name)
    return logger