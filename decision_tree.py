#!/usr/bin/env python3

from sys import exit, argv
import time
import pandas as pd
import numpy as np
import argparse

p=1.0

def id3(df, t, f):
    """
    ID3 Decision Tree generator.

    Parameter:
    df -- the dataframe of training data
    t -- target attribute
    f -- list of attributes

    Return:
    root -- the fully formed decision tree
    """
    root, ig = {}, {}  # root node, IG dict
    attr = df.columns.drop(t)  # get attribute set of df
    for a in attr:
        ig[a] = find_information_gain(df, t, a)  # find IG of attr
    highest_ig = max(ig, key=lambda key: ig[key])  # return key of highest val
    s = make_split(df, highest_ig)  # find splits on highest IG attr
    root = {highest_ig: {}}  # found root for further branches
    for v in s.keys():  # for each outcome of root
        df_branch = df.where(df[highest_ig] == v).dropna()  # remove root node
        # if entropy of potential branch is zero, all outcomes same = term leaf
        if find_entropy(df_branch[t]) == 0:
            # add leaf branch
            root[highest_ig][v] = s[v][t].value_counts().idxmax()
        else:  # otherwise branch has further subbranches = decision
            if len(attr) - 1 == 0:  # if no more attr to divide on
                # entropy not 0, next branch isn't pure, imperfect decision
                root[highest_ig][v] = s[v][t].value_counts().idxmax()
                return root
            else:  # if more attr to split on, can recurse
                # recurse on split, dropping root attr
                root[highest_ig][v] = id3(s[v].drop(highest_ig, axis=1), t, f)
    return root


def find_entropy(t):
    """
    Finds entropy of target attribute in training set.
    H(S) = \sum_{x\inX}{ -p(x)*log_2{p(x)} }

    Parameter:
    t -- target attribute

    Return:
    h -- entropy of target attribute
    """
    h = 0
    v, n = np.unique(t, return_counts=True)  # get values and distinct v
    for x in range(len(v)):
        px = n[x] / np.sum(n)
        h += -px * np.log2(px)
    return h


def find_information_gain(df, t, s):
    """
    Finds information gain of target attribute in training set.
    IG(S,A) = H(S) - \sum_{t\inT}{ p(t) * H(t)} = H(S) - H(S|A)

    Parameter:
    df -- the dataframe of training data
    t -- target attribute
    s -- splitting attribute
    """
    total_h = find_entropy(df[t])  # find entropy of entire system
    split_h = 0  # entropy after potential split
    v, n = np.unique(df[s], return_counts=True)  # get values and distinct v
    for x in range(len(v)):
        pt = n[x] / np.sum(n)
        split = df.where(df[s] == v[x]).dropna()[t]  # remove missing attrs
        split_h += pt * find_entropy(split)
    return total_h - split_h


def make_split(df, t):
    """
    Splits a dataframe on attribute.

    Parameter:
    df -- the dataframe to split
    t -- target attribute to split upon

    Return:
    new_df -- split dataframe
    """
    new_df = {}
    for df_key in df.groupby(t).groups.keys():
        new_df[df_key] = df.groupby(t).get_group(df_key)
    return new_df


def find_accuracy(dt, t):
    """
    Determines accuracy of the system.
    Accuracy = (1 - error) = (TP+TN)/(TP+TN+FP+FN)

    Parameter:
    dt -- the decision tree
    t -- a set of testing examples

    Return:
    accuracy -- how accurate the system is
    """
    correct, total = 0, 0
    for _, e in t.iterrows():
        total += 1  # TP+TN+FP+FN
        if e[len(e) - 1] == predict_decision(dt, e):
            correct += 1  # TP+TN
    return round(((correct / total) * 100), 1)


def predict_decision(dt, e, depth_limit, depth=0):
    """
    Predicts decision on a testing example.

    Parameter:
    dt -- the decision tree
    e -- a testing example

    Return:
    decision -- a classification/decision
    """
    split = list(dt.keys())[0]
    try:
        branch = dt[split][e[split]]
    except KeyError:
        # attribute not found in split
        return None, False
    if not isinstance(branch, dict):  # terminal leaf node/decision
        return branch, True
    if(depth + 1 < depth_limit):
        return predict_decision(branch, e, depth_limit, 1 + depth)  # recurse into sub-dict
    else:
        return branch, False


def holdout(df):
    """
    Splits a dataframe of examples into training and testing data.

    Parameter:
    df -- a dataframe of examples
    p -- proportion of training vs testing (0.00..1.00]

    Return:
    train -- training examples
    test -- testing examples
    """
    d = df.copy()
    train = d.sample(frac=p)  # split, and randomize
    test = d.drop(train.index)  # remove train data from df
    return train, test


def count_leaves(dt, c=[0, 0]):
    """
    Count number of non-leaf and leaf branches.

    Parameter:
    dt -- the decision tree
    c -- a counter

    Return:
    c -- a count for both non-leeaves and leaves
    """
    c[0] += 1
    leaves = dt.keys()
    for leaf in leaves:
        branches = dt[leaf].values()
        for branch in branches:
            if isinstance(branch, dict):
                count_leaves(branch, c)
            else:
                c[1] += 1
    return c


def print_tree(dt, indent=0):
    """
    Prints decision tree in a better fashion.

    Parameter:
    dt -- the tree to display
    indent -- used internally for indentation
    """
    for key, value in dt.items():
        print("  " * indent + str(key))
        if isinstance(value, dict):  # if subdict
            print_tree(value, indent + 1)
        else:  # otherwise value
            print("  " * (indent + 1) + str(value))


def print_statistics(dt, t, tr, te, trs, tes):
    """
    Prints diagnostics regarding decision tree.

    Parameter:
    dt -- the decision tree
    t -- the time it took to generate dt
    tr -- classification ability of training data
    te -- classification ability of novel (test) data
    trs -- number of training examples
    tes -- number of testing examples
    """
    s, d = count_leaves(dt)  # splits and decisions
    print(f"Using {trs} training examples and {tes} testing examples.")
    print(f"Tree contains {s} non-leaf nodes and {d} leaf nodes.")
    print("Took {:.2f} seconds to generate.".format(t))
    print(f"Was able to classify {tr}% of training data.")
    print(f"Was able to classify {te}% of testing data.\n")


def load_csv(f):
    """
    Loads CSV file into pandas dataframe.
    .CSV file is organized such that decision is the last column and features
    are other columns. The first column is the name of decision and features.

    An example .CSV might be:
    F1 F2 F3 F4 F5 F6 D
     0  0  1  1  0  1 1
     1  0  1  1  0  1 0
     1  0  0  0  1  1 0
     1  1  0  1  1  0 1

    Where F1..Fn are attributes and D is the decision

    Parameter:
    f -- the filename for the .CSV file

    Return:
    df -- a dataframe of examples
    """
    df = pd.read_csv(f)  # open file as parse CSV into dataframe

    print(f"{f} was successfully loaded.")
    return df


def get_data(data_file):
    """
    Load CSV data depending on holdout or not.

    Return:
    train -- a set of training examples
    test -- a set of testing examples
    """
    train, test = holdout(data_file)
    return train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = 'dataset-1.csv', help = "Input csv file name of dataset")
    args = parser.parse_args()
    dataset = args.dataset
    train, test = get_data(dataset)
    decision_name = train.columns[len(train.columns) - 1]
    start_time = time.time()
    dt = id3(train, decision_name, train.columns[:-1])  # get decision tree
    end_time = time.time();
    t = end_time - start_time
    tr_size = len(train)
    te_size = len(test)
    tr_ability = find_accuracy(dt, train)
    te_ability = find_accuracy(dt, test)
    print_statistics(dt, t, tr_ability, te_ability, tr_size, te_size)
    exit(0)
