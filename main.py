import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from utils.logger import create_logger


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Geometric Graph Foundation Model')



configs = parser.parse_args()