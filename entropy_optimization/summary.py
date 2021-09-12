import sys

import matplotlib.pyplot as plt
import numpy as np


class Tee:
    def __init__(self, fname):
        self.stdout = sys.stdout
        self.file = open(fname, "a")

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class Summary:
    def __init__(self, fname, rewards, **stats):
        self.fname = fname
        self.runs = len(rewards)
        self.horizon = len(rewards[0])
        self.sort_rewards = rewards
        for i in range(self.runs):
            for t in range(self.horizon):
                rewards[i][t].sort()
        self.spearmanrs = stats['spearmanr']
        self.numidx = stats['numidx']
        self.numint = None
        if 'numint' in stats.keys():
            self.numint = stats['numint']

    def printout(self):
        for i in range(self.runs):
            print('Run {}:'.format(i + 1))
            if not self.numint:
                print('\tRound\tAverage\tMax\tMin\tTop10\tTop20\trho\tNumInt')
            else:
                print('\tRound\tAverage\tMax\tMin\tTop10\tTop20\trho\tNumInt\tTotalInt')
            rewards = self.sort_rewards[i]
            for t in range(self.horizon):
                Average = np.average(rewards[t])
                Max = rewards[t][-1]
                Min = rewards[t][0]
                Top10 = np.average(rewards[t][-10:])
                Top20 = np.average(rewards[t][-20:])
                nidx = self.numidx[i][t]
                if t == 0:
                    sr = 'None'
                    if not self.numint:
                        print('\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}'.format(t, Average, Max, Min, Top10,
                                                                                            Top20, sr, nidx))
                    else:
                        print('\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t5000'.format(t, Average, Max, Min,
                                                                                                  Top10,
                                                                                                  Top20, sr, nidx))
                else:
                    sr = self.spearmanrs[i][t - 1]
                    if not self.numint:
                        print('\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}'.format(t, Average, Max, Min,
                                                                                                Top10, Top20, sr, nidx))
                    else:
                        nint = self.numint[i][t - 1]
                        print(
                            '\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}'.format(t, Average, Max, Min,
                                                                                                  Top10,
                                                                                                  Top20, sr, nidx,
                                                                                                  nint))

    def plotit(self):
        plt.figure(figsize=(10, 3))

        average = np.array(self.sort_rewards).mean(axis=2)
        ax = plt.subplot(1, 2, 1)
        ax.grid()
        ax.plot([t + 1 for t in range(self.horizon)], average.T, '.:')
        ax.plot([t + 1 for t in range(self.horizon)], average.mean(axis=0), '*-k', label='Average across runs',
                linewidth=2)
        ax.legend()
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Average rewards')
        ax.set_title('Average rewards')  # fontweight='bold')
        ax.axis('tight')
        ax.set_xticks([t + 1 for t in range(self.horizon)])

        top5 = np.array(self.sort_rewards)[:, :, -5:].mean(axis=2)
        ax = plt.subplot(1, 2, 2)
        ax.grid()
        ax.plot([t + 1 for t in range(self.horizon)], top5.T, '.:')
        ax.plot([t + 1 for t in range(self.horizon)], top5.mean(axis=0), '*-k', label='Average across runs',
                linewidth=2)
        ax.legend()
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Average top 5 rewards')
        ax.set_title('Average top 5 rewards')  # fontweight='bold')
        ax.axis('tight')
        ax.set_xticks([t + 1 for t in range(self.horizon)])

        plt.tight_layout()
        plt.savefig(self.fname)
