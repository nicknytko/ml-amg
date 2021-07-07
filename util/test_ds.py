import numpy as np
import numpy.linalg as la
import pickle
import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import sys
import argparse
import scipy.sparse as sp
import os

sys.path.append(os.path.dirname(os.getcwd()))
import ns.lib.helpers as helpers
from ns.model import *

ds = MeshDataset()
print(len(ds))
print(ds[0])
