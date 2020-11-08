import helper
import copy
from matplotlib import rcParams
import sys, getopt
from itertools import combinations
import os
import fim

LOGGING = True
LOG_FI_FILE = ''
LOG_AR_FILE = ''

# count support
def get_support(item, tracts):
  supp = 0
  for t in tracts:
    if set(item).issubset(t):
      supp += 1
  return supp/len(tracts)

def get_confidence(s, i, fi):
  items = [f for (f, p) in fi]
  supports = [p for (f, p) in fi]

  return supports[items.index(i[0])]/supports[items.index(s)]

def get_frequent_candidates(candidates, tracts, threshold):
  # prune elements violating downward closure 
  if LOGGING: helper.log(LOG_FI_FILE, "pruning with min. support %.2f..." % threshold)
  freq_cand = []
  for c in candidates:
    supp = get_support(c, tracts) 
    if LOGGING: helper.log(LOG_FI_FILE, "support for %s is %f" % (c, supp))
    if supp >= threshold:
      if LOGGING: helper.log(LOG_FI_FILE, 'removing %s' % c)
      freq_cand.append((c, supp))

  return freq_cand

# returns powerset of a given set
def powerset(s):
  if len(list(s)) > 0:
    return [set(x) for l in range(1, len(list(s))+1) for x in combinations(list(s), l)]
  return {}

def get_new_candidates(prev_candidates):
  new_candidates = []
  if len(prev_candidates) == 0: return []
  k = len(prev_candidates[0]) # level/length of candidates

  # try every pair of itemsets
  for i in range(len(prev_candidates)): 
    for j in range(i+1, len(prev_candidates)):
      # avoid duplicates: all merge the itemsets with shared items
      skip = False
      new_cand = {}
      for x in range(k-1):
        if prev_candidates[i][x] != prev_candidates[j][x]:
          skip = True
          break
      if not skip:
        new_cand = set(prev_candidates[i]).union(set(prev_candidates[j]))

      # enforce downward closure: check that all subsets of previous level are frequent
      if new_cand != {}:
        #print("from %s and %s we get %s " % (str(prev_candidates[i]), str(prev_candidates[j]), str(new_cand)))
        skip = False
        all_subsets = [c for c in powerset(new_cand) if len(list(c)) == k] # get all subsets of prev level
        for sub in all_subsets:
          if sorted(list(sub)) not in prev_candidates:
            skip = True
            break
        if not skip:
          sset = sorted(list(new_cand))
          new_candidates.extend([sset])
          
  return new_candidates

def apriori(tracts, threshold):
  if LOGGING: helper.log(LOG_FI_FILE, "Running Apriori algorithm...")
  # items making up dataset
  items = set.union(*map(set, tracts))
  # generate singleton sets from these items
  candidates = [[[i] for i in items]] # change to numpy
  fi = []
  k = 0 
  while len(candidates[k])>0:
    if LOGGING: helper.log(LOG_FI_FILE, "Candidates of length %d: %s " % (k+1, str(candidates[k])))
    freq_cand = get_frequent_candidates(candidates[k], tracts, threshold)
    if LOGGING: helper.log(LOG_FI_FILE, "Frequent itemsets of length %d: %s" % (k+1, str(freq_cand)))
    if freq_cand != []: fi.append(freq_cand)
    cand_alone = [c for (c, s) in freq_cand]
    new_candidates = get_new_candidates(cand_alone)
    candidates.append(new_candidates)
    k+=1

  return [i for l in fi for i in l]

def get_association_rules(fi, conf_thres):
  rules = []
  for i in fi:
    # generate asso rule
    subsets = [sorted(list(s)) for s in powerset(set(i[0])) if len(list(s)) > 2 and len(list(s)) < len(list(i[0]))]
    # compute confidence
    for s in subsets:
      conf = get_confidence(s, i, fi)
      if  conf >= conf_thres:
        rules.append([s, set(i[0]).difference(s), conf])

  return rules

def save_asso_rules(a_r):
  with open(LOG_AR_FILE, 'w') as af:
    for r in a_r:
      af.write(str(r[0]) + "=>" + str(r[1]) + ": " + str(r[2]) + "\n")
    
if __name__ == "__main__":
  #get parameters
  try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:s:c:")
  except getopt.GetoptError:
    print("Provide the following input: -d <dataset> -s <support> -c <confidence>")
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print("Provide the following input: -d <dataset> -s <support> -c <confidence>")
      sys.exit()
    elif opt == "-d":
      which = arg
    elif opt == "-s":
      supp = arg
    elif opt == "-c":
      conf = arg
  
  #check dataset exists
  if which not in helper.DATASETS:
    print("Unknown setup (%s)!" % which)
    exit()

  #check dataset exists
  try:
    conf = float(conf)
    if conf <= 0 or conf > 1:
      print("Confidence needs to be between 0 and 1.")
      exit()
  except AttributeError:
    raise Exception('Confidence needs to be a float')
  try:
    supp = float(supp)
    if supp <= 0 or supp > 1:
      print("Support needs to be between 0 and 1.")
      exit()
  except AttributeError:
    raise Exception('Support needs to be a float')
  
  try:
    method_load =  eval("helper.load_%s" % helper.DATASETS[which]["format"])
  except AttributeError:
    raise Exception("No known method to load this data type (%)!" % helper.DATASETS[which]["format"])

  ## logging code
  LOG_FILE = "output/apriori/" + which + "_" + str(supp) + ".txt"
  LOG_AR_FILE = "output/asso_r/" + which + "_" + str(supp) + "_" + str(conf) + ".txt"
  # erase previous runs of same dataset + support combo
  helper.init_file(LOG_FILE) 
  helper.init_file(LOG_AR_FILE)

  tracts, U = method_load(helper.DATASETS[which]["in_file"], helper.DATASETS[which]["sep"])

  apriori_fi = apriori(tracts, float(supp))
  eclat_fi = fim.eclat(tracts, target='s', supp=float(supp)*100, report='s')

  print("apriori ", len(apriori_fi))
  print("ecalt ", len(eclat_fi))

  a_r = get_association_rules(apriori_fi, conf)
  print("a_r ", len(a_r))
  save_asso_rules(a_r)
  
  if LOGGING: helper.log(LOG_FI_FILE, "========== Apriori: Frequent itemsets of %s at threshold %s are: %s" % (which, supp, str(apriori_fi)))
  if LOGGING: helper.log(LOG_FI_FILE, "========== Eclat: Frequent itemsets of %s at threshold %s are: %s" % (which, supp, str(eclat_fi)))