import numpy as np

A_dgp = np.loadtxt("./data/A_dgp.txt")
mu_dgp = np.loadtxt("./data/mu_dgp.txt").reshape((-1, 1))
mu_target = np.loadtxt("./data/mu_target.txt").reshape((-1, 1))


def real_reward(h):
    res = mu_target - np.matmul(A_dgp, mu_dgp + h.T)
    return -np.dot(res.T, res).diagonal()
