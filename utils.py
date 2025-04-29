# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/12/19 
@Author  :   tensorgao 
@Version :   1.0
@Contact :   gaoqiang_mx@163.com
@Desc    :   None
'''

import torch


# utils.py

import logging

# Create and configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# If you don't want duplicate log entries, avoid adding the handler multiple times
logger.propagate = False


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)