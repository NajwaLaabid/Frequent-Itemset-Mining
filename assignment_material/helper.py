import sys, re, itertools
import numpy as np, scipy.sparse, scipy.stats as st
import matplotlib.pyplot as plt
import pdb
from matplotlib import rcParams
import fim
import pandas
import time
import os
import math

DATASETS = {}
DATASETS["bats"] = {"in_file": "datasets/bats/bats.mat", "format": "matrix", "sep": ","}
DATASETS["retail"] = {"in_file": "datasets/retail/retail.dat", "format": "trans_num", "sep": " "}
DATASETS["accidents"] = {"in_file": "datasets/accidents/accidents.dat", "format": "trans_num", "sep": " "}
DATASETS["groceries"] = {"in_file": "datasets/groceries/groceries.csv", "format": "trans_txt", "sep": ","}
DATASETS["abalone"] = {"in_file": "datasets/abalone/abalone.data", "format": "data_txt", "sep": ","}
DATASETS["house"] = {"in_file": "datasets/house/house.data", "format": "data_txt", "sep": ","}
DATASETS["adult"] = {"in_file": "datasets/adult/adult.data", "format": "data_txt", "sep": ","}
DATASETS["test"] = {"in_file": "datasets/test/test.dat", "format": "trans_num", "sep": " "}
DATASETS["random_U"] = {"in_file": "datasets/random/random_U.dat", "format": "trans_num", "sep": " "}
DATASETS["random_B"] = {"in_file": "datasets/random/random_B.dat", "format": "trans_num", "sep": " "}
DATASETS["random_H"] = {"in_file": "datasets/random/random_H.dat", "format": "trans_num", "sep": " "}
DATASETS["random_V"] = {"in_file": "datasets/random/random_V.dat", "format": "trans_num", "sep": " "}
DATASETS["random_P"] = {"in_file": "datasets/random/random_P.dat", "format": "trans_num", "sep": " "}

RANDOM = {}
RANDOM["U"] = {"type": "uniform", "ntrans": 1000, "nitems": 50, "density": 0.1}
RANDOM["B"] = {"type": "blocks", "blocks": [(100,5)]*10}

RANDOM["P"] = {"type": "pieces", "ntrans": 1000, "nitems": 50, "nb": 11, "H": 100, "W": 5, "overlapH": 0.25, "overlapW": 0.2}

RANDOM["H"] = {"type": "horizontal", "ntrans": 1000, "nitems": 50, "density": 0.1, "nb": 5, "D": 10}
RANDOM["V"] = {"type": "vertical", "ntrans": 1000, "nitems": 50, "density": 0.1, "nb": 5, "D": 200}

def make_random_matrix(params):
    print(params)
    if params["type"] == "uniform":
        X = np.random.random((params["ntrans"], params["nitems"])) < params["density"]
    elif params["type"] == "blocks":
        X = make_random_blocks(params["blocks"])
    elif params["type"] == "pieces":
        X = make_random_pieces(params["ntrans"], params["nitems"], params["nb"], params["H"], params["W"], params.get("overlapH", 0), params.get("overlapW", 0))
    elif params["type"] == "horizontal":
        X = make_random_horizontal(params["ntrans"], params["nitems"], params["nb"], params["density"], params["D"])
    elif params["type"] == "vertical":
        X = make_random_horizontal(params["nitems"], params["ntrans"], params["nb"], params["density"], params["D"]).T
    else:
        X = np.zeros((params["ntrans"], params["nitems"]), dtype=bool)
    return X

def make_random_blocks(blocks):
    lR, lC = zip(*blocks)
    X = np.zeros((sum(lR), sum(lC)), dtype=bool)
    cumsumR, cumsumC = (0, 0)
    for b in blocks:
        X[cumsumR:cumsumR+b[0], cumsumC:cumsumC+b[1]] = 1 
        cumsumR += b[0]
        cumsumC += b[1]
    return X
    
def make_random_pieces(totH, totW, nb, H, W, overlapH, overlapW):
    X = np.zeros((totH, totW), dtype=bool)
    Hi, Wi = (0, 0)
    for i in range(nb):
        X[Hi:Hi+H, Wi:Wi+W] = 1
        if Wi+W > totW:
            X[Hi:Hi+H, :Wi+W-totW] = 1
            if Hi+H > totH:
                X[:Hi+H-totH, :Wi+W-totW] = 1
        elif Hi+H > totH:                
            X[:Hi+H-totH, Wi:Wi+W] = 1
        Hi = int((Hi+(1-overlapH)*H) % totH)
        Wi = int((Wi+(1-overlapW)*W) % totW)
    return X

def make_random_horizontal(totH, totW, nb, density, W):
    X = np.zeros((totH, totW), dtype=bool)
    H = int((density * totH * totW)/(nb*W))
    for i in range(nb):
        Wi = W*i % totW
        ids = np.random.choice(totH, H, replace=False)
        X[ids, Wi:Wi+W] = 1
        if Wi+W > totW:
            X[ids, :Wi+W-totW] = 1
    return X
            
def load_trans_num(in_file, sep=" "):
    print("in_file ", in_file)
    tracts = [1, 2, 3]
    # with open(in_file) as fp:
    #     for line in fp:
    #         l = line.split(sep)
    #         if '' in l: print(l)
    with open(in_file) as fp:
        tracts = [frozenset([int(s.strip()) for s in line.strip().split(sep) if not s == '']) for line in fp if not re.match("#", line)]
    U = sorted(set().union(*tracts))
    return tracts, U

def load_trans_txt(in_file, sep=" "):
    with open(in_file) as fp:
        tracts = [frozenset([s.strip() for s in line.strip().split(sep)]) for line in fp if not re.match("#", line)]
    U = sorted(set().union(*tracts))
    map_items = dict([(v,k) for (k,v) in enumerate(U)])
    tracts = [frozenset([map_items[i] for i in t]) for t in tracts]
    return tracts, U

def load_matrix(in_file, sep=" "):
    with open(in_file) as fp:
        firstline = fp.readline()
        parts = firstline.strip().strip("#").split(sep)
        try:
            s = [k for (k,v) in enumerate(parts) if int(v) != 0]
            U = range(len(s))
            tracts = [frozenset(s)]
        except ValueError:
            U = parts
            tracts = []
        tracts.extend([frozenset([k for (k,v) in enumerate(line.strip().split(sep)) if int(v) != 0]) for line in fp if not re.match("#", line)])
    if len(U) <= max(set().union(*tracts)):
        print("Something went wrong while reading!")
        print(U)
        print(max(set().union(*tracts)))
        return [], []
    return tracts, U

def load_sparse_num(in_file, sep=" "):
    tracts = {}
    with open(in_file) as fp:
        for line in fp:
            if not re.match("#", line):
                i, j = map(int, line.strip().split(sep)[:2])
                if i not in tracts:
                    tracts[i] = []
                tracts[i].append(j)
    U = range(max(set().union(*tracts.values()))+1)
    tracts = [frozenset(tracts.get(ti, [])) for ti in range(max(tracts.keys())+1)]
    return tracts, U

def load_sparse_txt(in_file, sep=" "):
    tracts = {}
    with open(in_file) as fp:
        for line in fp:
            if not re.match("#", line):
                i, j = line.strip().split(sep)[:2]
                i = int(i)
                if i not in tracts:
                    tracts[i] = []
                tracts[i].append(j)
    U = sorted(set().union(*tracts.values()))
    map_items = dict([(v,k) for (k,v) in enumerate(U)])
    tracts = [frozenset([map_items[i] for i in tracts.get(ti,[])]) for ti in range(max(tracts.keys())+1)]
    return tracts, U

def load_data_txt(in_file, sep=" "):
    bin_file = ".".join(in_file.split(".")[:-1]+["bininfo"])
    bininfo, U = read_bininfo(bin_file)
    #print(bininfo)
    tracts = []
    with open(in_file) as fp:
        for line in fp:
            if not re.match("#", line):
                parts = line.strip().split(sep)
                t = []
                for k, part in enumerate(parts):
                    if k in bininfo:
                        if "bool" in bininfo[k]:
                            t.append(bininfo[k]["offset"]+1*(part in bininfo[k]["bool"]))
                        elif "cats" in bininfo[k]:
                            if part in bininfo[k]["cats"]:
                                t.append(bininfo[k]["offset"]+bininfo[k]["cats"].index(part))
                        elif "bounds" in bininfo[k]:
                            off = 0
                            v = float(part)
                            while off < len(bininfo[k]["bounds"]) and v > bininfo[k]["bounds"][off]:
                                off += 1
                            t.append(bininfo[k]["offset"]+off)
                tracts.append(frozenset(t))
    #U = sorted(set().union(*tracts.values()))
    return tracts, U

def read_bininfo(bin_file):
    bininfo = {}
    with open(bin_file) as fp:
        for line in fp:
            tmp = re.match("^(?P<pos>[0-9]+) *(?P<name>[^ ]+) *(?P<type>\w+) *(?P<quote>[\'\"])(?P<details>.*)(?P=quote)", line)
            if tmp is not None:
                if tmp.group("type") == "BOL":
                    bininfo[int(tmp.group("pos"))] = {"bool": tmp.group("details").split(","), "name": tmp.group("name")}
                elif tmp.group("type") == "CAT":
                    bininfo[int(tmp.group("pos"))] = {"cats": tmp.group("details").split(","), "name": tmp.group("name")}
                else:
                    tt = re.search("equal\-(?P<type>(width)|(height)) *k=(?P<k>[0-9]+)", tmp.group("details"))
                    if tt is not None:
                        bininfo[int(tmp.group("pos"))] = {"type": "equal-%s" % tt.group("type"), "k": int(tt.group("k")), "name": tmp.group("name")}
                    else:
                        try:
                            bininfo[int(tmp.group("pos"))] = {"type": "fixed", "bounds": sorted(map(float, tmp.group("details").split(","))), "name": tmp.group("name")}
                        except ValueError:
                            pass

    fields = []
    ks = sorted(bininfo.keys())
    for k in ks:
        bininfo[k]["offset"] = len(fields)
        if "bool" in bininfo[k]:
            fields.extend(["%s_%s" % (bininfo[k]["name"], v) for v in ["False", "True"]])
        elif "cats" in bininfo[k]:
            fields.extend(["%s_%s" % (bininfo[k]["name"], v) for v in bininfo[k]["cats"]])
        elif "bounds" in bininfo[k]:
            fields.append("%s_:%s" % (bininfo[k]["name"], bininfo[k]["bounds"][0]))
            fields.extend(["%s_%s:%s" % (bininfo[k]["name"], bininfo[k]["bounds"][i], bininfo[k]["bounds"][i+1]) for i in range(len(bininfo[k]["bounds"])-1)])
            fields.append("%s_%s:" % (bininfo[k]["name"], bininfo[k]["bounds"][-1]))
        elif "k" in bininfo[k]:
            fields.extend(["%s_bin%d" % (bininfo[k]["name"], v) for v in range(bininfo[k]["k"])])
    return bininfo, fields
                        
def save_trans_num(tracts, U, out_file, sep=" "):
    print("Writing to %s, %d transactions, %d items\n# %s" % (out_file, len(tracts), len(U), ",".join(["%s" % i for i in U])))
    with open(out_file, "w") as fo:
        fo.write("\n".join([sep.join(["%d" % i for i in sorted(t)]) for t in tracts]))
        
def save_sparse_num(tracts, U, out_file, sep=" "):
    print("Writing to %s (sparse), %d transactions, %d items\n# %s" % (out_file, len(tracts), len(U), ",".join(["%s" % i for i in U])))
    with open(out_file, "w") as fo:
        fo.write("\n".join(["\n".join(["%d%s%d" % (ti, sep, i) for i in t]) for ti, t in enumerate(tracts) if len(t) > 0]))

def filter_trans(tracts, keep_items):
    map_items = dict([(v,k) for (k,v) in enumerate(keep_items)])
    keep_tids, ftracts = zip(*[(ti, frozenset([map_items[i] for i in t if i in map_items])) for ti, t in enumerate(tracts) if len(t.intersection(keep_items)) > 0])
    return keep_tids, ftracts
        
def trans_to_array(tracts, U, sparse=False):
    if sparse:
        if isinstance(U[0], str): col = len(U)
        else: col = max(U)
        M = scipy.sparse.lil_matrix((len(tracts),col+1),dtype=bool)
    else:
        M = np.zeros((len(tracts),len(U)),dtype=bool)
    for ti,t in enumerate(tracts):
        M[ti, list(t)] = 1
    return M

def array_to_trans(M):
    trans = []
    U = list(range(M.shape[1]))
    for r in M:
        trans.append(frozenset(np.where(r)[0]))
    return trans, U

def init_file(f):
  if os.path.exists(f): open(f, 'w').close()

def log(logf, text):
  with open(logf, 'a') as lf:
    lf.write(text+"\n")

def plot_mat(M, U, out_file):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.spy(M)
    ax.set_aspect('auto')
    plt.xticks(range(len(U)), U, rotation='vertical')
    plt.tick_params(axis='x', which='major', labelsize=3)
    plt.savefig(out_file)
    plt.close()

def plot_hist(data, out_file, weights=None):
    plt.figure()
    plt.hist(data, weights=weights)
    plt.savefig(out_file)
    plt.close()

def parse_describe(data):
    results = {}

    results['n_obs'], (results['min'], results['max']), results['avg'], results['variance'], skew, kurtosis = st.describe(data)

    results['n_obs'] = format(results['n_obs'], '.0f') if not math.isnan(results['n_obs']) else 0
    results['min'] = format(results['min'], '.0f') if not math.isnan(results['min']) else 0
    results['max'] = format(results['max'], '.0f') if not math.isnan(results['max']) else 0
    results['avg'] = round(results['avg'], 4) if not math.isnan(results['avg']) else 0.0
    results['variance'] = round(results['variance'], 4) if not math.isnan(results['variance']) else 0.0

    return results

def describe_dataset(dataset, tracts, mat, plotHists=False):
    items = [i for t in tracts for i in t]
    lengths = [len(t) for t in tracts]
    results = {}

    ## number of items and transactions, min, max, avg
    results['items'], min_max_i, mean_i, var_i, skew_i, kurtosis_i = st.describe(items)
    results['trans'], (results['min_trans'], results['max_trans']), results['avg_trans'], var_l, skew_l, kurtosis_l = st.describe(lengths)

    ## casting to int
    results['items'] = len(set(items))
    results['trans'] = format(results['trans'], '.0f')

    ## rounding
    results['avg_trans'] = round(results['avg_trans'], 4)

    ## density
    results['density'] = round(mat.getnnz()/np.prod(mat.shape), 4)

    ## histograms
    if plotHists:
        plot_hist(items, out_file='plots/histograms/'+dataset+'_items.png', weights=np.ones(len(items))/len(items))
        plot_hist(lengths, out_file='plots/histograms/'+dataset+'_trans.png', weights=np.ones(len(lengths))/len(lengths))
    
    return results