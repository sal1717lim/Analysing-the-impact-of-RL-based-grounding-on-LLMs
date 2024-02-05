'''
PPO implementation taken from https://github.com/openai/spinningup
'''

import numpy as np
import scipy
from scipy import signal
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import re
from tqdm import tqdm
import json
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def question_parser(file):
    data=json.load(open(file))
    test=[]
    for q in data["questions"]:
        qst=q["question"]+"the option :"+str(q["options"])
        test.append((qst,))
    for q in data["scenarios"]:
        qst="we have this trajectory:"+str(q["actions"])+" "+q["question"]
        test.append((qst,))
    return test