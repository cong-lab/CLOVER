import numpy as np
from numpy.linalg import inv
from scipy.stats import spearmanr

from feature import get_feature
from reward import real_reward
from search import local_search


class Algorithm:
    """
    A subclass of Algorithm implements a sequential decision making algorithm.
    Subclass should implement the following:
    - estimate()
    - update()
    """

    def __init__(self, bsz, intervention, feature, reward):
        self.bsz = bsz
        idx = np.random.choice(range(len(intervention)), bsz, replace=False)
        self.picked_interventions = [intervention[idx]]
        self.picked_rewards = [reward[idx]]
        self.x = feature[idx]
        self.y = reward[idx]
        self.idx = list(idx)
        self.numidx = [len(self.idx)]
        self.spearmanr = []

    def estimate(self):
        raise NotImplementedError

    def update(self, intervention, feature, reward):
        raise NotImplementedError


class Offline(Algorithm):
    """
    Offline sequential selection
    """

    def __init__(self, bsz, intervention, feature, reward, lbd):
        super(Offline, self).__init__(bsz, intervention, feature, reward)
        self.lbd = lbd
        self.theta = None

    def estimate(self):
        _, p = self.x.shape
        v = np.dot(self.x.T, self.x) + self.lbd * np.eye(p)
        iv = inv(v)
        self.theta = np.dot(iv, np.dot(self.x.T, self.y - np.mean(self.y)))

    def update(self, intervention, feature, reward):
        self.estimate()
        values = np.dot(feature, self.theta).reshape(-1)
        idx = np.argsort(values)[- self.bsz:]
        self.picked_interventions.append(intervention[idx])
        self.picked_rewards.append(reward[idx])
        new_idx = [i for i in idx if i not in self.idx]
        self.idx += new_idx
        self.x = np.append(self.x, feature[new_idx], axis=0)
        self.y = np.append(self.y, reward[new_idx], axis=0)
        self.spearmanr.append(spearmanr(values, reward.reshape(-1)).correlation)
        self.numidx.append(len(self.idx))


class Online(Algorithm):
    """
    Online sequential selection
    """

    def __init__(self, bsz, intervention, feature, reward, lbd, c):
        super(Online, self).__init__(bsz, intervention, feature, reward)
        self.lbd = lbd
        self.c = c
        self.theta = None
        self.iv = None

    def estimate(self):
        _, p = self.x.shape
        v = np.dot(self.x.T, self.x) + self.lbd * np.eye(p)
        self.iv = inv(v)
        self.theta = np.dot(self.iv, np.dot(self.x.T, self.y - np.mean(self.y)))

    def update(self, intervention, feature, reward):
        self.estimate()
        values = (np.dot(feature, self.theta) + self.c * np.sqrt(
            np.dot(feature, np.dot(self.iv, feature.T)).diagonal().reshape((-1,)))).reshape(-1)
        idx = np.argsort(values)[-self.bsz:]
        self.picked_interventions.append(intervention[idx])
        self.picked_rewards.append(reward[idx])
        new_idx = [i for i in idx if i not in self.idx]
        self.idx += new_idx
        self.x = np.append(self.x, feature[new_idx], axis=0)
        self.y = np.append(self.y, reward[new_idx], axis=0)
        self.spearmanr.append(spearmanr(values, reward.reshape(-1)).correlation)
        self.numidx.append(len(self.idx))


class Prol(Algorithm):
    """
    Path-regularized online sequential selection
    """

    def __init__(self, bsz, intervention, feature, reward, lbd, c, transformer, n):
        super(Prol, self).__init__(bsz, intervention, feature, reward)
        self.lbd = lbd
        self.c = c
        self.theta = None
        self.iv = None
        self.transformer = transformer
        self.intervention = intervention
        self.feature = feature
        self.reward = reward
        self.n = n
        self.numint = []

    def estimate(self):
        _, p = self.x.shape
        v = np.dot(self.x.T, self.x) + self.lbd * np.eye(p)
        self.iv = inv(v)
        self.theta = np.dot(self.iv, np.dot(self.x.T, self.y - np.mean(self.y)))

    def update(self, intervention, feature, reward):
        self.estimate()
        values = (np.dot(feature, self.theta) + self.c * np.sqrt(
            np.dot(feature, np.dot(self.iv, feature.T)).diagonal().reshape((-1,)))).reshape(-1)
        sort_idx = np.argsort(values)
        idx = sort_idx[-self.bsz:]
        self.picked_interventions.append(intervention[idx])
        self.picked_rewards.append(reward[idx])
        new_idx = [i for i in idx if i not in self.idx]
        self.idx += new_idx
        self.x = np.append(self.x, feature[new_idx], axis=0)
        self.y = np.append(self.y, reward[new_idx], axis=0)
        self.spearmanr.append(spearmanr(values, reward.reshape(-1)).correlation)
        self.numidx.append(len(self.idx))

        top_idx = sort_idx[- self.bsz * 2:]
        top_intervention = intervention[top_idx]
        new_intervention = local_search(top_intervention, intervention, self.n)
        new_feature = get_feature(new_intervention)
        if self.transformer is not None:
            new_feature = self.transformer.fit_transform(new_feature)
        new_values = (np.dot(new_feature, self.theta) + self.c * np.sqrt(
            np.dot(new_feature, np.dot(self.iv, new_feature.T)).diagonal().reshape((-1,)))).reshape(-1)
        threshold = values[top_idx[self.bsz]]
        add_idx = []
        for i in range(len(new_intervention)):
            if new_values[i] >= threshold:
                add_idx.append(i)
        if add_idx:
            intervention = np.append(intervention, new_intervention[add_idx], axis=0)
            feature = np.append(feature, new_feature[add_idx], axis=0)
            tmp = real_reward(new_intervention[add_idx])
            reward = np.append(reward, tmp.reshape(-1), axis=0)
        self.numint.append(len(reward))
        return intervention, feature, reward
