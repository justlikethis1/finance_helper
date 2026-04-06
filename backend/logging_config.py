import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# 创建日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 日志格式配置
# 添加了文件名和行号信息，便于定位问题
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(threadName)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 设置根日志级别
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    # 输出到控制台
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 输出到文件，支持自动轮转
        RotatingFileHandler(
            os.path.join(LOG_DIR, "app.log"),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,  # 保留5个备份文件
            encoding="utf-8"
        )
    ]
)

# 创建不同模块的日志记录器
def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器"""
    return logging.getLogger(name)
