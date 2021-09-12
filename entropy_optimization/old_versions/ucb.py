#################################################
###### ucb module
#################################################

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from numpy.linalg import inv

def non_ovlp(times, c, T, bsize, pca_component, lbd, design, entropy, estimate_entropy, features, scaler, pca, low_design, low, path):
    """not allow for overlap across different batches"""
    
    average_all = [[] for t in range(T)]
    power2_all = [[] for t in range(T)]
    average_top = [[] for t in range(T)]
    power2_top = [[] for t in range(T)]
    regret_all = [[] for t in range(T)]
    regret_top = [[] for t in range(T)]
    
    for j in range(times):
        savepath = path+"_ucb_run-"+str(j)+".txt"
        file = open(savepath,"w")
        
        # step1 -initializing
        if low:
            try:
                Design = list(np.random.choice(low_design, bsize, replace = False))    # start with low-entropy designs 
            except:
                print('not enough design!')
        else:
            Design = list(np.random.choice(design, bsize, replace = False))
        Entropy = [estimate_entropy[d] for d in Design]
        X = np.array([features[d] for d in Design])
        Y = np.array(Entropy).reshape(-1, 1)
        design_chosen = Design
        t = 1
        file.write("______Round "+str(t)+"______\n")
        for d in Design:
            file.write(str(d)+"\n")
        sum_real_entropy = sum([entropy[d] for d in Design])
        average_all[t - 1].append(sum_real_entropy / bsize)
        power2_all[t - 1].append(int(2 ** (sum_real_entropy / bsize)))

        temp = list(entropy.values())
        temp.sort(reverse = True)
        s1 = sum(temp[0: bsize])
        s2 = sum(temp[0: 10])
        regret_all[t - 1].append((s1 - sum_real_entropy) / bsize)

        temp = [entropy[d] for d in Design]
        temp.sort(reverse = True)
        sum_real_entropy_top = sum(temp[0: 10])
        average_top[t - 1].append(sum_real_entropy_top / 10)
        power2_top[t - 1].append(int(2 ** (sum_real_entropy_top / 10)))

        regret_top[t - 1].append((s2 - sum_real_entropy_top) / 10)

        while t < T:
            t += 1
            # step2 -regression
            X_1 = scaler.transform(pca.transform(X))
            V = np.dot(X_1.T, X_1) + lbd * np.eye(pca_component)
            iV = inv(V)
            theta = np.dot(iV, np.dot(X_1.T, Y))

            # step3 - calculate score for each x that was not included in the previous batches
            score = {}
            keys = [d for d in design if d not in design_chosen]
            x = np.array([features[d] for d in keys])
            x = scaler.transform(pca.transform(x))
            values = list((np.dot(x, theta) + c * np.sqrt(np.dot(x, np.dot(iV, x.T)).diagonal().reshape(-1,1))).reshape(-1,))
            score = dict(zip(keys,values))
            sorted_score = sorted(score.items(), key = lambda kv: kv[1], reverse = True)

            # step4 - pick the new batch
            new_design = [d[0] for d in sorted_score[0: bsize]]
    
            Design = new_design
            Entropy = [estimate_entropy[d] for d in Design]
            file.write("______Round "+str(t)+"______\n")
            for d in Design:
                file.write(str(d)+"\n")
            sum_real_entropy = sum([entropy[d] for d in Design])
            average_all[t - 1].append(sum_real_entropy / bsize)
            power2_all[t - 1].append(int(2 ** (sum_real_entropy / bsize)))
            
            temp = [entropy[d] for d in entropy.keys() if d not in design_chosen]
            temp.sort(reverse = True)
            s1 = sum(temp[0: bsize])
            s2 = sum(temp[0: 10])
            regret_all[t - 1].append((s1 - sum_real_entropy) / bsize)

            temp = [entropy[d] for d in Design]
            temp.sort(reverse = True)
            sum_real_entropy_top = sum(temp[0: 10])
            average_top[t - 1].append(sum_real_entropy_top / 10)
            power2_top[t - 1].append(int(2 ** (sum_real_entropy_top / 10)))

            regret_top[t - 1].append((s2 - sum_real_entropy_top) / 10)

            design_chosen = design_chosen + new_design

            if t < T:
                # step5 -renew X & Y
                X_add = np.array([features[d] for d in Design])
                Y_add = np.array(Entropy).reshape(-1, 1)
                X = np.append(X, X_add, axis = 0)
                Y = np.append(Y, Y_add, axis = 0)
        file.close()
                
    return (average_all, power2_all, average_top, power2_top, regret_all, regret_top)
    

def ovlp(times, c, T, bsize, pca_component, lbd, design, entropy, estimate_entropy, features, scaler, pca, x, low_design, low, path):
    """not allow for overlap across different batches"""
    
    average_all = [[] for t in range(T)]
    power2_all = [[] for t in range(T)]
    average_top = [[] for t in range(T)]
    power2_top = [[] for t in range(T)]
    regret_all = [[] for t in range(T)]
    regret_top = [[] for t in range(T)]
    overlap = [[] for t in range(T - 1)]
    
    for j in range(times):
        savepath = path+"_ucb_run-"+str(j)+".txt"
        file = open(savepath,"w")
        
        # step1 -initializing
        if low:
            try:
                Design = list(np.random.choice(low_design, bsize, replace = False))    # start with low-entropy designs
            except:
                print('not enough design!')
        else:
            Design = list(np.random.choice(design, bsize, replace = False))
        Entropy = [estimate_entropy[d] for d in Design]
        X = np.array([features[d] for d in Design])
        Y = np.array(Entropy).reshape(-1, 1)
        design_chosen = Design
        t = 1
        file.write("______Round "+str(t)+"______\n")
        for d in Design:
            file.write(str(d)+"\n")
        sum_real_entropy = sum([entropy[d] for d in Design])
        average_all[t - 1].append(sum_real_entropy / bsize)
        power2_all[t - 1].append(int(2 ** (sum_real_entropy / bsize)))

        temp = list(entropy.values())
        temp.sort(reverse = True)
        s1 = sum(temp[0: bsize])
        s2 = sum(temp[0: 10])
        regret_all[t - 1].append((s1 - sum_real_entropy) / bsize)

        temp = [entropy[d] for d in Design]
        temp.sort(reverse = True)
        sum_real_entropy_top = sum(temp[0: 10])
        average_top[t - 1].append(sum_real_entropy_top / 10)
        power2_top[t - 1].append(int(2 ** (sum_real_entropy_top / 10)))

        regret_top[t - 1].append((s2 - sum_real_entropy_top) / 10)

        while t < T:
            t += 1
            # step2 -regression
            X_1 = scaler.transform(pca.transform(X))
            V = np.dot(X_1.T, X_1) + lbd * np.eye(pca_component)
            iV = inv(V)
            theta = np.dot(iV, np.dot(X_1.T, Y))

            # step3 - calculate score for each x that was not included in the previous batches
            score = {}
            values = list((np.dot(x, theta) + c * np.sqrt(np.dot(x, np.dot(iV, x.T)).diagonal().reshape(-1, 1))).reshape(-1,))
            score = dict(zip(design, values))
            sorted_score = sorted(score.items(), key=lambda kv: kv[1], reverse=True)

            # step4 - pick the new batch
            new_design = [d[0] for d in sorted_score[0:bsize]]

            Design = new_design
            Entropy = [estimate_entropy[d] for d in Design]
            file.write("______Round "+str(t)+"______\n")
            for d in Design:
                file.write(str(d)+"\n")
            overlap_design = [d for d in new_design if d in design_chosen]
            overlap[t - 2].append(len(overlap_design))
            design_chosen = design_chosen + new_design
            average_real_entropy = sum([entropy[d] for d in Design]) / bsize
            average_all[t - 1].append(average_real_entropy)
            power2_all[t - 1].append(int(2 ** average_real_entropy))

            temp = list(entropy.values())
            temp.sort(reverse=True)
            s1 = sum(temp[0:bsize]) / bsize
            s2 = sum(temp[0:10]) / 10
            regret_all[t - 1].append(s1 - average_real_entropy)

            temp = [entropy[d] for d in Design]
            temp.sort(reverse=True)
            average_real_entropy_top = sum(temp[0:10]) / 10
            average_top[t - 1].append(average_real_entropy_top)
            power2_top[t - 1].append(int(2 ** average_real_entropy_top))

            regret_top[t - 1].append(s2 - average_real_entropy_top)

            if t < T:
                # step5 -renew X & Y
                X_add = np.array([features[d] for d in Design])
                Y_add = np.array(Entropy).reshape(-1, 1)
                X = np.append(X, X_add, axis=0)
                Y = np.append(Y, Y_add, axis=0)
        file.close()        
    
    return (average_all, power2_all, average_top, power2_top, regret_all, regret_top, overlap)
