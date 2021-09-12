#################################################
###### plot1t module
#################################################

import matplotlib.pyplot as plt
import numpy as np


def plot4tuning(path, rs, average, power, regret, overlap, bench_overlap, ovlp, pca_component, lbd):
    """plot used while tuning"""
    
    def aveplot():
        """plot of average entropy in each round""" 
        
        average_all = average["average_all"]
        bench_average_all = average["bench_average_all"]
        T = len(average_all[0])
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title("average of selected entropy")
        for i, r in enumerate(rs):
            ax = plt.subplot(3, 4, i + 1)
            ax.grid()
            ax.plot([t + 1 for t in range(T)], average_all[i], ".:")
            ax.plot(
                [t + 1 for t in range(T)],
                np.array(average_all[i]).mean(axis=-1),
                "*-k",
                label="average across 10 runs",
                linewidth=2,
            )
            ax.scatter(
                [t + 2 for t in range(T - 1)],
                [np.mean(bench_average_all[t]) for t in range(T - 1)],
                marker="^",
                c="b",
                label="benchmark",
            )
            ax.legend()
            ax.set_xlabel("rounds")
            ax.set_ylabel("average of selected entropy")
            ax.set_title("diameter of ellipsoid=%3.2f" % r)  # fontweight='bold')
            ax.axis("tight")
            ax.set_xticks([t + 1 for t in range(T)])
            ax.set_ylim(0,7)
        plt.tight_layout()
        plt.savefig(path + "/%d_%.2f_average.pdf" % (pca_component, lbd))


    def p2aveplot():
        """plot of 2 to the power of average entropy in each round"""
        
        power2_all = power["power2_all"]
        bench_power2_all = power["bench_power2_all"]
        T = len(power2_all[0])
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title("number of different barcodes")
        for i, r in enumerate(rs):
            ax = plt.subplot(3, 4, i + 1)
            # ymin, ymax = 2300, 3800
            ax.grid()
            ax.plot([t + 1 for t in range(T)], power2_all[i], ".:")
            ax.plot(
                [t + 1 for t in range(T)],
                np.array(power2_all[i]).mean(axis=-1),
                "*-k",
                label="average across 10 runs",
                linewidth=2,
            )
            ax.scatter(
                [t + 2 for t in range(T - 1)],
                [np.mean(bench_power2_all[t]) for t in range(T - 1)],
                marker="^",
                c="b",
                label="benchmark",
            )
            ax.legend()
            ax.set_xlabel("rounds")
            ax.set_ylabel("~2^{average of selected entropy}")
            ax.set_title("diameter of ellipsoid=%3.2f" % r)  # fontweight='bold')
            ax.axis("tight")
            ax.set_xticks([t + 1 for t in range(T)])
            ax.set_ylim(1,128)
        plt.tight_layout()
        plt.savefig(path + "/%d_%.2f_power2.pdf" % (pca_component, lbd))
 
 
    def regretplot():
        """regret plot"""
        
        regret_all = regret["regret_all"]
        bench_regret_all = regret["bench_regret_all"]
        T = len(regret_all[0])
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title("instantaneous regret of selected entropy")
        for i, r in enumerate(rs):
            ax = plt.subplot(3, 4, i + 1)
            # ymin, ymax = 2300, 3800
            ax.grid()
            ax.plot([t + 1 for t in range(T)], regret_all[i], ".:")
            ax.plot(
                [t + 1 for t in range(T)],
                np.array(regret_all[i]).mean(axis=-1),
                "*-k",
                label="average across 10 runs",
                linewidth=2,
            )
            ax.scatter(
                [t + 2 for t in range(T - 1)],
                [np.mean(bench_regret_all[t]) for t in range(T - 1)],
                marker="^",
                c="b",
                label="benchmark",
            )
            ax.legend()
            ax.set_xlabel("rounds")
            ax.set_ylabel("instantaneous regret of selected entropy")
            ax.set_title("diameter of ellipsoid=%3.2f" % r)  # fontweight='bold')
            ax.axis("tight")
            ax.set_xticks([t + 1 for t in range(T)])
            ax.set_ylim(0,7)
        plt.tight_layout()
        plt.savefig(path + "/%d_%.2f_regret.pdf" % (pca_component, lbd))
    
    
    def topaveplot():
        """top average plot"""
        
        average_top = average["average_top"]
        bench_average_top = average["bench_average_top"]
        T = len(average_top[0])
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title("average of top 10 selected entropy")
        for i, r in enumerate(rs):
            ax = plt.subplot(3, 4, i + 1)
            # ymin, ymax = 2300, 3800
            ax.grid()
            ax.plot([t + 1 for t in range(T)], average_top[i], ".:")
            ax.plot(
                [t + 1 for t in range(T)],
                np.array(average_top[i]).mean(axis=-1),
                "*-k",
                label="average across 10 runs",
                linewidth=2,
            )
            ax.scatter(
                [t + 2 for t in range(T - 1)],
                [np.mean(bench_average_top[t]) for t in range(T - 1)],
                marker="^",
                c="b",
                label="benchmark",
            )
            ax.legend()
            ax.set_xlabel("rounds")
            ax.set_ylabel("average of top 10 selected entropy")
            ax.set_title("diameter of ellipsoid=%3.2f" % r)  # fontweight='bold')
            ax.axis("tight")
            ax.set_xticks([t + 1 for t in range(T)])
            ax.set_ylim(0,7)
        plt.tight_layout()
        plt.savefig(path + "/%d_%.2f_topaverage.pdf" % (pca_component, lbd))
     
     
    def topp2aveplot():
        """top power 2 average plot"""
        
        power2_top = power["power2_top"]
        bench_power2_top = power["bench_power2_top"]
        T = len(power2_top[0])
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title("top number of different barcodes")
        for i, r in enumerate(rs):
            ax = plt.subplot(3, 4, i + 1)
            # ymin, ymax = 2300, 3800
            ax.grid()
            ax.plot([t + 1 for t in range(T)], power2_top[i], ".:")
            ax.plot(
                [t + 1 for t in range(T)],
                np.array(power2_top[i]).mean(axis=-1),
                "*-k",
                label="average across 10 runs",
                linewidth=2,
            )
            ax.scatter(
                [t + 2 for t in range(T - 1)],
                [np.mean(bench_power2_top[t]) for t in range(T - 1)],
                marker="^",
                c="b",
                label="benchmark",
            )
            ax.legend()
            ax.set_xlabel("rounds")
            ax.set_ylabel("~2^{average of top 10 selected entropy}")
            ax.set_title("diameter of ellipsoid=%3.2f" % r)  # fontweight='bold')
            ax.axis("tight")
            ax.set_xticks([t + 1 for t in range(T)])
            ax.set_ylim(1,128)
        plt.tight_layout()
        plt.savefig(path + "/%d_%.2f_toppower2.pdf" % (pca_component, lbd))
        
        
    def topregret():
        """topregret plot"""
        
        regret_top = regret["regret_top"]
        bench_regret_top = regret["bench_regret_top"]
        T = len(regret_top[0])
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title("instantaneous regret of top 10 selected entropy")
        for i, r in enumerate(rs):
            ax = plt.subplot(3, 4, i + 1)
            # ymin, ymax = 2300, 3800
            ax.grid()
            ax.plot([t + 1 for t in range(T)], regret_top[i], ".:")
            ax.plot(
                [t + 1 for t in range(T)],
                np.array(regret_top[i]).mean(axis=-1),
                "*-k",
                label="average across 10 runs",
                linewidth=2,
            )
            ax.scatter(
                [t + 2 for t in range(T - 1)],
                [np.mean(bench_regret_top[t]) for t in range(T - 1)],
                marker="^",
                c="b",
                label="benchmark",
            )
            ax.legend()
            ax.set_xlabel("rounds")
            ax.set_ylabel("instantaneous regret of top 10 selected entropy")
            ax.set_title("diameter of ellipsoid=%3.2f" % r)  # fontweight='bold')
            ax.axis("tight")
            ax.set_xticks([t + 1 for t in range(T)])
            ax.set_ylim(0,7)
        plt.tight_layout()
        plt.savefig(path + "/%d_%.2f_topregret.pdf" % (pca_component, lbd))


    def overlapplot():
        """overlap plot"""
        
        T = len(overlap[0]) + 1
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title("number of overlap")
        for i, r in enumerate(rs):
            ax = plt.subplot(3, 4, i + 1)
            ax.grid()
            ax.plot([t + 2 for t in range(T - 1)], overlap[i], ".:")
            ax.plot(
                [t + 2 for t in range(T - 1)],
                np.array(overlap[i]).mean(axis=-1),
                "*-k",
                label="average across 10 runs",
                linewidth=2,
            )
            ax.scatter(
                [t + 2 for t in range(T - 1)],
                [np.mean(bench_overlap[t]) for t in range(T - 1)],
                marker="^",
                c="b",
                label="benchmark",
            )
            ax.legend()
            ax.set_xlabel("rounds")
            ax.set_ylabel("number of overlap")
            ax.set_title("diameter of ellipsoid=%3.2f" % r)  # fontweight='bold')
            ax.axis("tight")
            ax.set_xticks([t + 2 for t in range(T - 1)])
        plt.tight_layout()
        plt.savefig(path + "/%d_%.2f_overlap.pdf" % (pca_component, lbd))
    
    aveplot()
    p2aveplot()
    regretplot()
    topaveplot()
    topp2aveplot()
    topregret()
    if ovlp:
        overlapplot()


def plot1pair(path, average, power, regret, overlap, bench_overlap, ovlp, pca_component, lbd):
    """plot 1 pair of parameters"""
    
    average_all = average["average_all"]
    bench_average_all = average["bench_average_all"]
    power2_all = power["power2_all"]
    bench_power2_all = power["bench_power2_all"]
    regret_all = regret["regret_all"]
    bench_regret_all = regret["bench_regret_all"]
    average_top = average["average_top"]
    bench_average_top = average["bench_average_top"]
    power2_top = power["power2_top"]
    bench_power2_top = power["bench_power2_top"]
    regret_top = regret["regret_top"]
    bench_regret_top = regret["bench_regret_top"]
    T = len(average_all)
    
    plt.clf()
    plt.figure(figsize = (16, 6))

    ax = plt.subplot(2, 3, 1)
    ax.grid()
    ax.plot([t + 1 for t in range(T)], average_all, '.:')
    ax.plot([t + 1 for t in range(T)], np.array(average_all).mean(axis=-1), '*-k', label='Average across 10 runs', linewidth=2)
    ax.scatter([t + 2 for t in range(T-1)], [np.mean(bench_average_all[t]) for t in range(T-1)], marker = '^', c = 'b', label = 'benchmark')
    ax.legend()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Average of selected entropy')
    ax.set_title('Average of selected entropy') #fontweight='bold')
    ax.axis('tight')
    ax.set_xticks([t + 1 for t in range(T)])

    ax = plt.subplot(2, 3, 2)
    ax.grid()
    ax.plot([t + 1 for t in range(T)], power2_all, '.:')
    ax.plot([t + 1 for t in range(T)], np.array(power2_all).mean(axis=-1), '*-k', label='Average across 10 runs', linewidth=2)
    ax.scatter([t + 2 for t in range(T-1)], [np.mean(bench_power2_all[t]) for t in range(T-1)], marker = '^', c = 'b', label = 'benchmark')
    ax.legend()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('~2^{Average of selected entropy}')
    ax.set_title('Number of different barcodes') #fontweight='bold')
    ax.axis('tight')
    ax.set_xticks([t + 1 for t in range(T)])

    ax = plt.subplot(2, 3, 3)
    ax.grid()
    ax.plot([t + 1 for t in range(T)], regret_all, '.:')
    ax.plot([t + 1 for t in range(T)], np.array(regret_all).mean(axis=-1), '*-k', label='Average across 10 runs', linewidth=2)
    ax.scatter([t + 2 for t in range(T-1)], [np.mean(bench_regret_all[t]) for t in range(T-1)], marker = '^', c = 'b', label = 'benchmark')
    ax.legend()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Regret of selected entropy')
    ax.set_title('Regret of selected entropy') #fontweight='bold')
    ax.axis('tight')
    ax.set_xticks([t + 1 for t in range(T)])

    ax = plt.subplot(2, 3, 4)
    ax.grid()
    ax.plot([t + 1 for t in range(T)], average_top, '.:')
    ax.plot([t + 1 for t in range(T)], np.array(average_top).mean(axis=-1), '*-k', label='Average across 10 runs', linewidth=2)
    ax.scatter([t + 2 for t in range(T-1)], [np.mean(bench_average_top[t]) for t in range(T-1)], marker = '^', c = 'b', label = 'benchmark')
    ax.legend()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Average of top selected entropy')
    ax.set_title('Average of top selected entropy') #fontweight='bold')
    ax.axis('tight')
    ax.set_xticks([t + 1 for t in range(T)])

    ax = plt.subplot(2, 3, 5)
    ax.grid()
    ax.plot([t + 1 for t in range(T)], power2_top, '.:')
    ax.plot([t + 1 for t in range(T)], np.array(power2_top).mean(axis=-1), '*-k', label='Average across 10 runs', linewidth=2)
    ax.scatter([t + 2 for t in range(T-1)], [np.mean(bench_power2_top[t]) for t in range(T-1)], marker = '^', c = 'b', label = 'benchmark')
    ax.legend()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('~2^{Average of top selected entropy}')
    ax.set_title('Number of top different barcodes') #fontweight='bold')
    ax.axis('tight')
    ax.set_xticks([t + 1 for t in range(T)])

    ax = plt.subplot(2, 3, 6)
    ax.grid()
    ax.plot([t + 1 for t in range(T)], regret_top, '.:')
    ax.plot([t + 1 for t in range(T)], np.array(regret_top).mean(axis=-1), '*-k', label='Average across 10 runs', linewidth=2)
    ax.scatter([t + 2 for t in range(T-1)], [np.mean(bench_regret_top[t]) for t in range(T-1)], marker = '^', c = 'b', label = 'benchmark')
    ax.legend()
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Regret of top selected entropy')
    ax.set_title('Regret of top selected entropy') #fontweight='bold')
    ax.axis('tight')
    ax.set_xticks([t + 1 for t in range(T)])

    plt.tight_layout()
    plt.savefig(path + '/%d_%.2f.pdf' %(pca_component,lbd))

    if ovlp:
        # check overlap
        plt.clf()
        plt.figure(figsize = (20, 10))
        plt.title('number of overlap')

        plt.grid()
        plt.plot([t + 2 for t in range(T-1)], overlap, '.:')
        plt.plot([t + 2 for t in range(T-1)], np.array(overlap).mean(axis=-1), '*-k', label='average across 10 runs', linewidth=2)
        plt.scatter([t + 2 for t in range(T-1)], [np.mean(bench_overlap[t]) for t in range(T-1)], marker = '^', c = 'b', label = 'benchmark')
        plt.legend()
        plt.xlabel('rounds')
        plt.ylabel('number of overlap')
        plt.title('overlap count')
        plt.xticks([t + 2 for t in range(T-1)])
        plt.savefig(path + '/%d_%.2f_overlap.pdf' %(pca_component,lbd))
    
