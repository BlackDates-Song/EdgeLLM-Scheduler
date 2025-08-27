import os
import argparse
import math
import csv
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

MINMAX = {
    "CPU":{"min":1.0,"max":5.0},
    "MEM":{"min":2.0,"max":16.0},
    "DELAY":{"min":5.0,"max":500.0},
    "LOAD":{"min":0.0,"max":1.0},
}

