import sys, re, itertools
import numpy as np, scipy.sparse, scipy.stats as st
import matplotlib.pyplot as plt
import pdb
from matplotlib import rcParams
import fim
import pandas
import time
import os
import helper

TARGETS = {}
TARGETS["maximal"] = 'm'
TARGETS["closed"] = 'c'
TARGETS["rule"] = 'r'

'''
    Returns statistics on all datasets considered.
'''
def get_all_stats(sets, out_file, plotMatrix=False, plotHists=False):
    df = pandas.DataFrame(columns=['dataset', 'items', 'trans', 'density', 'min_trans', 
                                    'max_trans', 'avg_trans'],
                          index=list(sets))

    for s in sets:
        print("computing stats for dataset: %s" % s)
        method_load =  eval("helper.load_%s" % (helper.DATASETS[s]["format"]))
        tracts, U = method_load(helper.DATASETS[s]["in_file"], helper.DATASETS[s]["sep"])
        M = helper.trans_to_array(tracts, U, sparse=True)
        df.loc[s] = pandas.Series(helper.describe_dataset(s, tracts, M, plotHists))
        df.loc[s]['dataset'] = s
        # plot binary matrix
        if plotMatrix:
            print("==== ploting binary matrix...")
            m_file = "plots/binaryM/"+s+".pdf"
            helper.plot_mat(M.toarray(), U, m_file)

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off') 
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

'''
    Reads the output of mining algorithms from corresponding files in algo_output, 
    and generates plots for the execution time over support for both algorithms, 
    in addition to a summary table of the main statistics (max., min., avg. and variance) 
    of the result set for each target pattern (closed/maximal itemsets and association rules).
'''
def analyze_mining_results(dataset, supports, algorithms, target, save_fi_summary=False, plot_runtime=False):
    if dataset == 'accidents': supports = range(40, 110, 10)

    ex_times = {'eclat': [], 'fpgrowth': []}

    df = pandas.DataFrame(columns=['support', 'n_obs', 'min', 'max', 'avg', 
                                    'variance'],
                          index=supports)
    for supp in supports:
        for algo in algorithms:
            #items = []
            #freqs = []
            lengths = []
            in_file = 'output/eclat_fpgrowth/' + dataset + '/' + target + '/' + algo.__name__ + '_' + str(supp) + '.txt'
            with open(in_file, 'r') as fp:
                content = fp.readlines()
                ex_times[algo.__name__].append(float(content[0]))
                for line in content[1:]:
                    data = line.split('  freq  ')
                    if target == 'rule': 
                        fi = data[0].split("  r  ")[1]
                    else:
                        fi = data[0].split(" ")
                    #freq = data[1].strip()
                    #items.extend(map(int, fi))
                    #freqs.append(float(freq))
                    lengths.append(len(fi))

        df.loc[supp]['support'] = supp
        if lengths != []: 
            df.loc[supp] = pandas.Series(helper.parse_describe(lengths))
            df.loc[supp]['support'] = supp
    
    ## save table of summary statistics of dataset mined with fpgrowth
    if save_fi_summary:
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off') 
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        fig.savefig("plots/fi_summary/fi_"+dataset+ "_" + target + ".pdf", bbox_inches='tight', pad_inches=0)

    ## plot execution time of fpgrowth and eclat for given dataset
    if plot_runtime:
        plt.figure()
        plt.plot(supports, ex_times['eclat'], color='r', label='eclat')
        plt.plot(supports, ex_times['fpgrowth'], color='b', label='fpgrowth')
        hline = np.array([0 for i in range(len(supports))])
        plt.plot(supports, hline, color='k', label='reference line')
        plt.legend(loc="upper right")
        plt.savefig("plots/runtime_plots/runtime_"+dataset+ "_" + target + ".png", bbox_inches='tight')
        plt.close()

'''
    Executes given mining algorithms on dataset over multiple support values, 
    and saves the output in a algo_support file under algo_output folder.
    Every output file starts with the execution time of the algorithm run,
    followed by the mined sets. Itemsets are saved as space separated integers, 
    followed by keyword 'freq' then the frequency of the set given as a fraction.
    Association rules are saved as antecedent + keyword 'r' + support of the association rule.
'''
def get_mining_results(dataset, supports, algorithms, target):
    if dataset == 'accidents': supports = range(40, 110, 10)
    repet = 2
    report = 's'
    if target == 'rule': report = 'x'

    ## test every algorithm on all values of support
    ## run every algo-support combination 'repet' times to get average estimates on execution time
    ## save the output of one run per algo-support to study it statistically
    for algo in algorithms:
        for supp in supports:
            ## get estimates on execution time
            ex_time = 0.
            for i in range(repet):
                start_time = time.time()
                fi = algo(tracts=tracts, target=TARGETS[target], report=report,supp=supp)
                #print("algo: ", algo, " support: ", supp, " rules: ", fi)
                ex_time += time.time() - start_time
            ex_time = round(ex_time/5, 4)
            out_file = 'output/eclat_fpgrowth/' + dataset + '/' + target + '/' + algo.__name__ + '_' + str(supp) + '.txt' 
            with  open(out_file, 'w') as of:
                of.write(str(ex_time)+'\n')
                if target == 'maximal' or target == 'closed':
                    of.write("\n".join('%s %s %s' % (' '.join(map(str, list(x[0]))), ' freq ', str(round(x[1], 4))) for x in fi))
                elif target == 'rule':
                    of.write("\n".join('%s %s %s %s %s' % (str(x[0]), ' r ', ' '.join(map(str, list(x[1]))), ' conf ', str(round(x[2], 4))) for x in fi))

if __name__ == "__main__":
    rcParams.update({'figure.autolayout': True})
    if len(sys.argv) < 2:
        print("Please choose a dataset: %s" % ", ".join(helper.DATASETS.keys()))
        exit()
    which = sys.argv[1]
    what = sys.argv[2]
    if re.match("all_datasets", which):
        sets = ["retail", "adult", "groceries", "house", "bats", "abalone", "accidents"]
        out_file = "plots/dataset_statistics.pdf"
        get_all_stats(sets, out_file, plotMatrix=True, plotHists=True)
        exit()
    if re.match("all_random", which):
        sets = ["random_U", "random_B", "random_H", "random_V", "random_P"]
        out_file = "plots/random_statistics.pdf"
        get_all_stats(sets, out_file, plotMatrix=True, plotHists=True)
        exit()
    if re.match("random", which):
        rwhich = "U"
        tmp = re.match("random_(?P<rwhich>\w+)", which)
        if tmp is not None and tmp.group("rwhich") in helper.RANDOM:
            rwhich = tmp.group("rwhich")
        M = helper.make_random_matrix(helper.RANDOM[rwhich])
        tracts, U = helper.array_to_trans(M)
        helper.save_trans_num(tracts, U, "datasets/random/random_%s.dat" % rwhich, " ")
    if which not in helper.DATASETS:
        print("Unknown setup (%s)!" % which)
        exit()
    if what not in TARGETS:
        print("Unknown mining target (%s)!" % what)
        exit()
    try:
        method_load =  eval("helper.load_%s" % helper.DATASETS[which]["format"])
    except AttributeError:
        raise Exception("No known method to load this data type (%)!" % helper.DATASETS[which]["format"])
    tracts, U = method_load(helper.DATASETS[which]["in_file"], helper.DATASETS[which]["sep"])

    print(tracts)
    print(U)
    # helper.save_trans_num(tracts, U, "X_%s.dat" % which, " ")
    M = helper.trans_to_array(tracts, U, sparse=True)
    # plot_mat(M.toarray(), U, which, folder='dataset')

    ## Mining 
    supports = np.linspace(50, 90, 10)
    supports = [round(s,2) for s in supports]
    algorithms = [fim.eclat, fim.fpgrowth]

    #print(fim.eclat(tracts, target='r', supp=0.1))

    get_mining_results(which, supports, algorithms, target=what)
    analyze_mining_results(which, supports, algorithms, what, save_fi_summary=True, plot_runtime=True)

