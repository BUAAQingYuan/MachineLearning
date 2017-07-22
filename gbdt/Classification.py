__author__ = 'PC-LiNing'

import math


class Binary_Classification_Loss:
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes.".format(self.__class__.__name__))

    def initialize(self,y_t,dataset):
        train_size = dataset.size
        for id in range(train_size):
            y_t[id] = 0.0

    def compute_gradient(self,dataset,subset,y_t, label_field='label'):
        next_gradient = {}
        for id in subset:
            try:
                label = dataset.instances[id][label_field]
            except KeyError:
                print(id)
            next_gradient[id] = 2.0 * label / (1+math.exp(2*label*y_t[id]))
        return next_gradient

    def update_y_t(self,y_t,tree,leaf_nodes,dataset,subset,lr):
        data_idset = set(range(dataset.size))
        subset = set(subset)
        # 对当前subset中的id更新y_t
        for node in leaf_nodes:
            for id in node.get_idset():
                y_t[id] += lr * node.get_weight()
        # 对不在subset中的id由tree得到weight
        for id in data_idset-subset:
            y_t[id] += lr * tree.get_predict_value(dataset.instances[id])
