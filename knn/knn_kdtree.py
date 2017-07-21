__author__ = 'PC-LiNing'

import csv
import numpy
from sklearn.cross_validation import train_test_split
from knn import KD_Tree


# load data
# apple = 0 , orange = 1, banana = 2
def loadDataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        X = numpy.zeros(shape=(len(dataset), 4),dtype=numpy.float32)
        Y = []
        count = 0
        for data in dataset:
            # label
            if data[4] == 'apple':
                data[4] = 0
            elif data[4] == 'orange':
                data[4] = 1
            elif data[4] == 'banana':
                data[4] = 2
            Y.append(data[4])
            # x
            for i in range(4):
                data[i] = float(data[i])
            X[count] = numpy.asarray(data[0:4],dtype=numpy.float32)
            count += 1
        Y = numpy.asarray(Y,dtype=numpy.int32)
        return X,Y


def get_target(label_set):
    counter ={}
    for i in label_set: counter[i] = counter.get(i, 0) + 1
    result = sorted([(freq,word) for word, freq in counter.items()],reverse=True)
    return result[0][1]


dataset, labels = loadDataset('testdata.txt')
X_train,X_test,Y_train,Y_test = train_test_split(dataset,labels,test_size=0.25)

kd = KD_Tree.KdTree(X_train.tolist(),Y_train.tolist())
nearest_num = 5
correct_num = 0
i = 0
for x in X_test:
    k_nearest = KD_Tree.find_k_nearest(kd, x, num=nearest_num)
    result = []
    for i_nearest in k_nearest:
         result.append(i_nearest.nearest_node.dom_label)
    target = get_target(result)
    if target == Y_test[i]:
        correct_num += 1
    i += 1

acc = float(correct_num / len(Y_test) * 100)
print("Test: "+str(len(Y_test)))
print("Correct: "+str(correct_num))
print("acc: "+str(acc))

