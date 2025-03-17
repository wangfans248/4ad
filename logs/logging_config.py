import logging
import os
from datetime import datetime


def setup_logging(log_filename=None):
    log_dir = "logs/Fanslogs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_filename is None:
        log_filename = os.path.join(
            log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    else:
        log_filename = os.path.join(log_dir, log_filename)

    # 使用 handlers 参数时，不要指定 filename 参数
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    print(f"日志已保存到 {log_filename}")
