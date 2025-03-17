import logging
import os
from datetime import datetime


def setup_logging(log_filename=None):
    log_dir = "logs/Fanslogs"
    if log_filename is None:
        log_filename = f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_path = os.path.join(log_dir, log_filename)

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    print(f"日志已保存到 {log_path}")