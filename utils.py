import logging
import os

from datetime import datetime


def avg(l: list):
    return int(sum(l)) / int(len(l))


def round_num(n):
    return "{:0.2f}".format(n)


def get_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


def get_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter(
        "%(asctime)s %(name)s - %(levelname)s - %(message)s")

    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")

    filename = f"logs/{log_file}"
    handler = logging.FileHandler(filename, "a", "utf-8")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
