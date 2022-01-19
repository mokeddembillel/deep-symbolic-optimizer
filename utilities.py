import math
import torch as T
import numpy as np
import torch.nn.functional as F
class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        # self.children = []
        self.data = data
