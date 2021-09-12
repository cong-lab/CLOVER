#################################################
###### entropy estimation module
#################################################

import os
import sys
import argparse
import math
import random

def cal_entropy(path, rnum):
    """calculate the 'entropy' using num of reads"""
    
    sam = []
    with open(path, "r") as file:
        sam = file.readlines()
        sam = [x.strip() for x in sam]
    N = len(sam) - 2
    if N < rnum:
        print("not enough reads")
    idx = [i + 3 for i in range(N)]
    random.shuffle(idx)
    idx = idx[0:rnum]
    count = 0
    d = {}
    for i in sam:
        count += 1
        design = ""
        if count > 2:
            line = str(i).strip("\n").split("\t")
            cigar = line[5]
            design = line[2]
            if count in idx:
                if cigar not in d.keys():
                    d[cigar] = 1
                else:
                    d[cigar] += 1
    name = design
    entropy = 0
    for j in d.keys():
        if d[j] > 0 and j != "None":
            entropy += -math.log(d[j] / rnum, 2) * d[j] / rnum    
    entropy += (len(d) - 1) / 200 # miller correction
    return (design, entropy)


def entropy_sample(path, rnum):
    """return the estimated entropy calculated from rnum of reads"""
    
    entropy = {}
    for file in os.listdir(path):
        if ".needleall" in file:
            (d, e) = cal_entropy(os.path.join(path, file), rnum)
            entropy[d] = e
    return entropy


