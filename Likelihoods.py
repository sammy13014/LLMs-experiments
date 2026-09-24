import numpy as np, pandas as pd, matplotlib.pyplot as plt


K = 3
STATES = [0, 1, 2]
TRUE = 0
LABELS = list("ABC")

Lx={0: np.array([0.50, 0.30, 0.20]),
      1: np.array([0.50, 0.30, 0.20]),    
      2: np.array([0.10, 0.20, 0.70])}

Ly = {0: np.array([0.60, 0.30, 0.10]),
      1: np.array([0.10, 0.30, 0.60]),
      2: np.array([0.60, 0.30, 0.10])}

LIK = {1: Lx, 2: Ly}         
COORD = {1: "x", 2: "y"}

Lx_close={0: np.array([0.50, 0.30, 0.20]),
      1: np.array([0.50, 0.30, 0.20]),    
      2: np.array([0.40, 0.35, 0.25])}

Ly_close = {0: np.array([0.60, 0.30, 0.10]),
      1: np.array([0.50, 0.35, 0.15]),
      2: np.array([0.60, 0.30, 0.10])}

LIK_close = {1: Lx_close, 2: Ly_close}         
COORD_close = {1: "x", 2: "y"}

def sample_true_state(Lx,Ly):
    """Sample a state and an observation from the likelihoods Lx and Ly."""
    # Sample a state
    
    # Sample an observation from Lx and Ly based on the sampled state
    obs_x = np.random.choice(LABELS, p=Lx[0])
    obs_y = np.random.choice(LABELS, p=Ly[0])
    
    return obs_x, obs_y

def kl_divergence(p, q):
    """Compute the Kullback-Leibler divergence between two probability distributions."""
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
