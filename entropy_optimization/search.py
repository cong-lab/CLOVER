from copy import deepcopy

import numpy as np


def local_search(top_intervention, intervention, n=10):
    l = len(top_intervention[0])
    assert n < l
    new_intervention = []
    for i in top_intervention:
        idx = np.random.choice(range(l), n, replace=False)
        idx1 = np.random.choice(range(l), n, replace=False)
        idx2 = np.random.choice(range(l), n, replace=False)
        for j in range(n):
            tmp = deepcopy(i)
            tmp[idx[j]] = 1 - tmp[idx[j]]
            diff = intervention - tmp
            if 0 not in np.sum(diff * diff, axis=1):
                if not new_intervention:
                    new_intervention.append(tmp)
                else:
                    diff2 = np.array(new_intervention) - tmp
                    if 0 not in np.sum(diff2 * diff2, axis=1):
                        new_intervention.append(tmp)

            if idx1[j] != idx2[j]:
                tmp = deepcopy(i)
                tmp[idx1[j]] = 1 - tmp[idx1[j]]
                tmp[idx2[j]] = 1 - tmp[idx2[j]]
                diff = intervention - tmp
                if 0 not in np.sum(diff * diff, axis=1):
                    if not new_intervention:
                        new_intervention.append(tmp)
                    else:
                        diff2 = np.array(new_intervention) - tmp
                        if 0 not in np.sum(diff2 * diff2, axis=1):
                            new_intervention.append(tmp)
    return np.array(new_intervention)
