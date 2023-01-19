import math

data = []
p = 0.8 # training data percentage from the dataset

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

    for i in range(train_size):
        d = euclidian_dist(train_set[i], new_sample)
        dists[i] = d

    k_neighbors = sorted(dists, key=dists.get)[:K]

    qty_label1, qty_label2 = 0, 0
    for index in k_neighbors:
        if train_set[index][-1] == 1:
            qty_label1 += 1
        else:
            qty_label2 += 1

    if qty_label1 > qty_label2:
        return 1
    else:
        return 2

if __name__ == "__main__":
    load_data('haberman.data')
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

