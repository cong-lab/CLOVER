#################################################
###### online learning using ucb
#################################################

from sklearn import preprocessing
from sklearn.decomposition import PCA
import entrprep
import benchmark
from ucb import non_ovlp as model    # not allow for overlap across different batches
import threading
import plot1t
import numpy as np

def main():

    # parameters
    T = 10    # horizon
    bsize = 96    # batch size
    nread = 100    # number of reads used to estimate entropy
    pca_component = 300    # number of PCs
    lbd = 10    # l_2 penalty
    times = 20    # number of running times 
    c = 4    # size of confidence ellipsoid
    
    # store designs in design, features and entropy into features and entropy
    design = []
    features = {}
    entropy = {}    # actual entropy    
    X_full = np.loadtxt('/labs/congle/PRT/Hiseq_20190725_TATS/sample_demultiplexing_10BC/al_ft_downsample0823/entropy_analysis_R300/data/data_fuzzy_encode/TATS_v2_D14_pDOX_fuzzy-3_X.txt')
    Y_full = np.loadtxt('/labs/congle/PRT/Hiseq_20190725_TATS/sample_demultiplexing_10BC/al_ft_downsample0823/entropy_analysis_R300/data/data_fuzzy_encode/TATS_v2_D14_pDOX_fuzzy-3_labels.txt')
    with open('/labs/congle/PRT/Hiseq_20190725_TATS/sample_demultiplexing_10BC/al_ft_downsample0823/entropy_analysis_R300/data/raw/TATS_v2_D14_pDOX.txt', 'r') as file:
        content = file.readlines()
        content = [x.strip() for x in content]
        for i in range(len(content)):
            temp = content[i].split('\t')[0]
            features[temp] = X_full[i, : ]
            entropy[temp] = Y_full[i]
            design.append(temp)
    low_design = [d for d in design if entropy[d] < 2]    # store the low-entropy designs


    # estimate entropy using nread reads 
    estimate_entropy = entrprep.entropy_sample('/labs/congle/PRT/Hiseq_20190725_TATS/sample_demultiplexing_10BC/al_ft_downsample0823/3_downsample-2/R300/TATS_v2_D14_pDOX', nread)


    # compute pca and standardizer
    pca = PCA(n_components = pca_component)
    pca.fit(X_full)
    X_pca = pca.transform(X_full)
    scaler = preprocessing.StandardScaler().fit(X_pca)
    
    
    # benchmark
    (bench_average_all, bench_power2_all, bench_average_top, bench_power2_top, bench_regret_all, bench_regret_top) = benchmark.non_ovlp(T, bsize, pca_component, lbd, design, entropy, estimate_entropy, features, scaler, pca, low_design, True, times)


    # ucb
    path = "../result/picked_design"    # path to save the picked designs
    (average_all, power2_all, average_top, power2_top, regret_all, regret_top) = model(times, c, T, bsize, pca_component, lbd, design, entropy, estimate_entropy, features, scaler, pca, low_design, True, path)       
    for t in range(T):
        print('Round %d average: mean = %.2f, std = %.2f' %(t + 1, np.mean(average_all[t]), np.std(average_all[t])))


    # plot it
    path = "../plot10"
    average = {"average_all":average_all, "bench_average_all":bench_average_all, "average_top":average_top, "bench_average_top":bench_average_top}
    power = {"power2_all":power2_all, "bench_power2_all":bench_power2_all, "power2_top":power2_top, "bench_power2_top": bench_power2_top}
    regret = {"regret_all":regret_all, "bench_regret_all":bench_regret_all, "regret_top": regret_top, "bench_regret_top": bench_regret_top}
    plot1t.plot1pair(path, average, power, regret, [], [], False, pca_component, lbd)


if __name__ == '__main__':
    main()
