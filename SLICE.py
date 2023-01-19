import math
import pandas as pd
import decision_tree as id3tree

data = []
p = 0.8 # training data percentage from the dataset
depth_limit = 2
def get_entropy(prob):
    if prob == 0 or prob == 1:
        return 0
    return prob*math.log2(1/prob)

#Import the data and append it to the list
def load_data(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            atributes = line.strip('\n').split(',')
            i = 1
            sub_data = []
            for x in atributes:
                if(i % 4 == 0):
                    sub_data.append(int(x))
                elif (i % 4 == 1):
                    sub_data.append(int(x)//5)
                elif (i % 4 == 2):
                    sub_data.append(int(x) // 10)
                elif (i % 4 == 3):
                    sub_data.append(int(x) //4)
                i += 1
            data.append(sub_data)

# information of the data set
def info_dataset(data, verbose=True):
    label1, label2 = 0, 0
    data_size = len(data)
    for datum in data:
        if datum[-1] == 1:
            label1 += 1
        else:
            label2 += 1
    if verbose:
        print('Total of samples: %d' % data_size)
        print('Total label 1: %d' % label1)
        print('Total label 2: %d' % label2)
    return [len(data), label1, label2]

# function to calculate euclidian distance
def euclidian_dist(p1, p2):
    dim, sum_ = len(p1), 0
    for index in range(dim - 1):
        sum_ += math.pow(p1[index] - p2[index], 2)
    return math.sqrt(sum_)

# calculates the euclidian distnces, sorts the data based on the distances and fetchs the k nearest neighbors. returns most recurring label.
def knn(train_set, new_sample, K):
    dists, train_size = {}, len(train_set)
    new_ex= []
    new_ex.append(new_sample)
    ex = pd.DataFrame(new_ex, columns=['F1', 'F2', 'F3', 'label'])
    for i in range(train_size):
        d = euclidian_dist(train_set[i], new_sample)
        dists[i] = d
    k_neighbors = sorted(dists, key=dists.get)[:K]
    qty_label1, qty_label2 = 0, 0
    small_qty_label1, small_qty_label2 = 0, 0
    new_data = []
    half_new_data = []
    count = 1
    for index in k_neighbors:
        new_data.append(train_set[index])
        if(count <= K//2):
            half_new_data.append(train_set[index])
            if train_set[index][-1] == 1:
                small_qty_label1 += 1
            else:
                small_qty_label2 += 1

        if train_set[index][-1] == 1:
            qty_label1 += 1
        else:
            qty_label2 += 1
        count += 1
    count = K//2
    Large_prop1 = qty_label1/K
    Large_prop2 = 1 - Large_prop1
    large_entropy = get_entropy(Large_prop1) + get_entropy(Large_prop2)
    small_prop1 = small_qty_label1/count
    small_prop2 = 1 - small_prop1
    small_entropy = get_entropy(small_prop1) + get_entropy(small_prop2)

    entropy_diff = large_entropy - small_entropy
    if(abs(entropy_diff) <= 0.1):
        print("HL aproximately equal to HS")
        df = pd.DataFrame(new_data, columns=['F1', 'F2', 'F3', 'label'])
        result = 0
        train, test = id3tree.get_data(df)
        decision_name = train.columns[len(train.columns) - 1]
        dt = id3tree.id3(train, decision_name, train.columns[:-1])  # get decision tree
        valid = False
        for _, e in ex.iterrows():
            result, valid = id3tree.predict_decision(dt, e, depth_limit)
        if(valid):
            return result
        else:
            if small_qty_label1 > small_qty_label2:
                return 1
            else:
                return 2
    if(entropy_diff > 0):
        print("HL is greater than HS")
        if small_qty_label1 > small_qty_label2:
            return 1
        else:
            return 2
    else:
        print("HS is greater than HL")
        df = pd.DataFrame(new_data, columns=['F1', 'F2', 'F3', 'label'])
        result = 0
        train, test = id3tree.get_data(df)
        decision_name = train.columns[len(train.columns) - 1]
        dt = id3tree.id3(train, decision_name, train.columns[:-1])  # get decision tree
        valid = False
        for _, e in ex.iterrows():
            result, valid = id3tree.predict_decision(dt, e, depth_limit)
            if (valid):
                return result
            else:
                if qty_label1 > qty_label2:
                    return 1
                else:
                    return 2

if __name__ == "__main__":
    load_data('haberman-1.data')
    _, label1, label2 = info_dataset(data, True)
    #Split the data set into train set and test set
    train_set, test_set = [], []
    max_label1, max_label2 = int(p * label1), int(p * label2)
    total_label1, total_label2 = 0, 0
    for sample in data:
        if (total_label1 + total_label2) < (max_label1 + max_label2):
            train_set.append(sample)
            if sample[-1] == 1 and total_label1 < max_label1:
                total_label1 += 1
            else:
                total_label2 += 1
        else:
            test_set.append(sample)

    print(test_set[0])
    print(knn(train_set, test_set[0], 12))

    # Counts the correct predictions of the test set with a given K
    correct, K = 0, 15
    for sample in test_set:
        label = knn(train_set, sample, K)
        if sample[-1] == label:
            correct += 1

    print("Train set size: %d" % len(train_set))
    print("Test set size: %d" % len(test_set))
    print("Correct predictions: %d" % correct)
    print("Accuracy: %.2f%%" % (100 * correct / len(test_set)))