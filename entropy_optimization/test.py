import argparse
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import algorithms
import summary
from feature import get_feature

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Sequential Experimental Design for Cell Reprogramming')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--times', type=int, default=5, help='number of repeating runs')
    parser.add_argument('--bsz', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--method', type=str, default='online', help='supported methods: offline, online, prol')
    parser.add_argument('--lbd', type=float, default=10)
    parser.add_argument('--radius', type=float, default=5, help='radius of the confidence ellipsoid')
    parser.add_argument('--transform', action='store_false')
    parser.add_argument('--npcs', type=int, default=100, help='number of principal components')
    parser.add_argument('--num', type=int, default=10, help='number of local searches')
    parser.add_argument('--output_dir', type=str, default='./output_online')
    args = parser.parse_args()

    # save printing results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    sys.stdout = summary.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = summary.Tee(os.path.join(args.output_dir, 'err.txt'))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}:{}'.format(k, v))

    # process data
    intervention = np.loadtxt('{}/intervention.txt'.format(args.data_dir))
    reward = np.loadtxt('{}/reward.txt'.format(args.data_dir))

    feature = get_feature(intervention)
    transformer = None
    if args.transform:
        transformer = Pipeline(steps=[('pca', PCA(n_components=args.npcs)), ('standardizer', StandardScaler())])
        feature = transformer.fit_transform(feature)

    # create saving arguments
    picked_interventions = []
    picked_rewards = []
    spearmanrs = []
    numidx = []
    if args.method == 'prol':
        numint = []

    # 
    for i in range(args.times):
        if args.method == 'offline':
            algorithm = algorithms.Offline(args.bsz, intervention, feature, reward, args.lbd)
        if args.method == 'online':
            algorithm = algorithms.Online(args.bsz, intervention, feature, reward, args.lbd, args.radius)
        if args.method == 'prol':
            algorithm = algorithms.Prol(args.bsz, intervention, feature, reward, args.lbd, args.radius, transformer,
                                        args.num)

        for t in range(args.horizon - 1):
            if args.method == 'prol':
                if t == 0:
                    intervention_new = intervention
                    feature_new = feature
                    reward_new = reward
                intervention_new, feature_new, reward_new = algorithm.update(intervention_new, feature_new, reward_new)
            else:
                algorithm.update(intervention, feature, reward)

        picked_interventions.append(algorithm.picked_interventions)
        picked_rewards.append(algorithm.picked_rewards)
        spearmanrs.append(algorithm.spearmanr)
        numidx.append(algorithm.numidx)
        if args.method == 'prol':
            numint.append(algorithm.numint)

    stats = {'spearmanr': spearmanrs, 'numidx': numidx}
    if args.method == 'prol':
        stats['numint'] = numint

    fname = None
    if args.method == 'offline':
        fname = os.path.join(args.output_dir, 'bsz{}_lbd{}_npcs{}.pdf'.format(args.bsz, args.lbd, args.npcs))
    if args.method == 'online':
        fname = os.path.join(args.output_dir,
                             'bsz{}_lbd{}_radius{}_npcs{}.pdf'.format(args.bsz, args.lbd, args.radius, args.npcs))
    if args.method == 'prol':
        fname = os.path.join(args.output_dir,
                             'bsz{}_lbd{}_radius{}_npcs{}_num{}.pdf'.format(args.bsz, args.lbd, args.radius, args.npcs,
                                                                            args.num))
    assert fname
    get_summary = summary.Summary(fname, picked_rewards, **stats)
    get_summary.printout()
    get_summary.plotit()
