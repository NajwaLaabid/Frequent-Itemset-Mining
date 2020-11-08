# Frequent Itemset Mining: eclat vs fp-growth

## Overview

Comparing the performance of two frequent itemset mining algorithms, [eclat](https://borgelt.net/eclat.html) and [fp-growth](https://github.com/enaeseth/python-fp-growth/blob/master/fp_growth.py) 
on 6 datasets:

* retail and accident ([source](http://fimi.uantwerpen.be/data/))
* groceries ([source](https://www.kaggle.com/irfanasrullah/groceries))
* bats ([source](https://www.european-mammals.org/))
* abalone ([source](https://archive.ics.uci.edu/ml/datasets/Abalone))
* house ([source](https://archive.ics.uci.edu/ml/datasets/congressional+voting+records))
* adult ([source](http://archive.ics.uci.edu/ml/datasets/Adult))

The study aims to identify the key characteristics of the datasets affecting the performance of the two algorithms. The report presents a summary of the key findings 
along with supporting figures.

## What's included

The repository includes:

* report.pdf: the report presenting key results and corresponding discussion
* /code: the code directory, including: 
    * /datasets: a copy of the dataset files used in the study.
    * /output: directory used to save the output of the two miners and related figures.
    * helper.py: includes helper functions for running the experiments.
    * miner.py: helper code for running the two mining algorithms, eclat and fp-growth.
    * main.py: main file to run the miners and generate figures.

You can clone the repository and run the file 'main.py' to re-execute the experiments. You can use the report as a reference for interpreting the results.
    
