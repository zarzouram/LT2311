import random
import numpy as np
import torch as th

# source: https://github.com/NVIDIA/framework-determinism/blob/master/pytorch.md

seed = 123
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
th.cuda.manual_seed_all(seed)
th.cuda.manual_seed(seed)
CUDA_LAUNCH_BLOCKING = 1
