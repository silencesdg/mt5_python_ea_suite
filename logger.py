import logging
import os

# 全局变量，用于存储logger实例
_logger_instance = None

def setup_logger(log_level="INFO"):
    global _logger_instance
    if _logger_instance:
        return _logger_instance

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "strategy.log")

    logger = logging.getLogger("StrategyLogger")
    
    # 设置日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    logger.setLevel(level_map.get(log_level, logging.INFO))

    # 防止重复添加handler
    if not logger.handlers:
        # 输出到文件
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level_map.get(log_level, logging.INFO))

        # 输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level_map.get(log_level, logging.INFO))

        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    _logger_instance = logger
    return logger

# 直接导出一个已经初始化好的logger实例
logger = setup_logger()
