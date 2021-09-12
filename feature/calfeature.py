#################################################
###### calculate features
#################################################

import numpy as np
from Levenshtein import jaro_winkler

# read target seq & length of designs
target = {}
g1_len = {}
g2_len = {}
with open("cut_design_ref_plus.txt", "r") as file:
    count = 0
    for line in file:
        count += 1
        if count % 2:
            key = line.strip()
        else:
            line = line.strip()
            target[key] = line[:-4]
            g1_len[key] = line[-4:-2]
            g2_len[key] = line[-2:]


# feature functions
def nctd(reference):
    """encode single nucleotide"""
    turn = {"A": "1 0 0 0 ", "T": "0 1 0 0 ", "C": "0 0 1 0 ", "G": "0 0 0 1 "}
    temp = []
    for i in range(54):
        temp.append(turn[reference[i + 3]])
    return "".join(temp)
    
def di_nctd(reference):
    """encode dinucleotide"""
    turn = [
        "AA",
        "AT",
        "AG",
        "AC",
        "TA",
        "TT",
        "TC",
        "TG",
        "GA",
        "GT",
        "GG",
        "GC",
        "CA",
        "CT",
        "CG",
        "CC",
    ]
    temp = ["0 " for i in range(16 * 53)]
    for i in range(53):
        idx = turn.index(reference[i + 3 : i + 5])
        temp[16 * i + idx] = "1 "
    return "".join(temp)

def MH_GC(reference):
    """encode micro homology and GC content"""
    GC = []
    MH = []
    for i in [k + 4 for k in range(52)]:
        for j in [k + i + 1 for k in range(56 - i)]:
            # deletion between i and j
            t = min(j - i, 15)
            if j + t <= 60:
                GC_1 = 1 - (
                    reference[i : i + t].count("G")
                    + reference[i : i + t].count("C")
                    + reference[j : j + t].count("G")
                    + reference[j : j + t].count("C")
                ) / (2 * t)
                GC.append(str(GC_1) + " ")
                MH.append(
                    str(jaro_winkler(reference[i : i + t], reference[j : j + t])) + " "
                )
            if i - t >= 0:
                GC_2 = 1 - (
                    reference[i - t : i].count("G")
                    + reference[i - t : i].count("C")
                    + reference[j - t : j].count("G")
                    + reference[j - t : j].count("C")
                ) / (2 * t)
                GC.append(str(GC_2) + " ")
                MH.append(
                    str(jaro_winkler(reference[i - t : i], reference[j - t : j])) + " "
                )
    return "".join(MH) + "".join(GC)


# save feature values
wfile = open("feature.txt", "w")
for key in target.keys():
    wfile.write(key + "\n")
    feature = (
        nctd(target[key])
        + di_nctd(target[key])
        + MH_GC(target[key])
        + g1_len[key]
        + " "
        + g2_len[key]
    )
    wfile.write(feature + "\n")
wfile.close()
