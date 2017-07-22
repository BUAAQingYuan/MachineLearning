__author__ = 'PC-LiNing'

from random import sample
from gbdt import Classification
from gbdt import Tree
import math
import datetime


class GBDT:
    def __init__(self, max_iter, sample_rate, learning_rate, max_depth, loss_type):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.lr = learning_rate
        self.max_depth = max_depth
        self.loss_type = loss_type
        self.loss = None
        self.trees = dict()
        self.y_t = dict()

    def compute_loss(self, dataset):
        loss = 0.0
        # loss对所有的训练样例进行计算
        for id in range(dataset.size):
            label = dataset.instances[id][dataset.label_field]
            y_id = self.y_t[id]
            p_1 = 1/(1+math.exp(-2.0*y_id))
            loss -= ((1 + label)*math.log(p_1)/2.0) + ((1 - label)*math.log(1-p_1)/2.0)
        return loss/dataset.size

    def train(self, dataset, train_data_ids):
        if self.loss_type == 'binary-classification':
            self.loss = Classification.Binary_Classification_Loss(n_classes=dataset.get_label_size())

        self.loss.initialize(self.y_t, dataset)
        for it in range(1,self.max_iter+1):
            subset = train_data_ids
            if 0 < self.sample_rate < 1:
                subset = sample(subset, int(len(subset) * self.sample_rate))
            # 计算负梯度
            gradient = self.loss.compute_gradient(dataset,subset,self.y_t)
            leaf_nodes = []
            tree = Tree.build_decision_tree(dataset,subset,gradient,0,leaf_nodes,self.max_depth,self.loss)
            self.trees[it] = tree
            # 更新 y_t
            self.loss.update_y_t(self.y_t,tree,leaf_nodes,dataset,subset,self.lr)
            # train loss
            train_loss = self.compute_loss(dataset)
            time_str = datetime.datetime.now().isoformat()
            print("{}: iter {}, loss {:g}".format(time_str, it, train_loss))

    def compute_y_t(self, instance):
        y_t = 0.0
        for it in range(1, self.max_iter+1):
            tree = self.trees[it]
            y_t += self.lr * tree.get_predict_value(instance)
        return y_t

    def predict(self,dataset, testset):
        predicts = []
        for i in range(len(testset)):
            instance = dataset.instances[testset[i]]
            y_t = self.compute_y_t(instance)
            p_1 = 1/(1+math.exp(-2.0*y_t))
            if p_1 > 0.5:
                predicts.append(1)
            else:
                predicts.append(-1)
        return predicts

    def compute_acc(self, dataset, testset):
        predicts = self.predict(dataset,testset)
        correct_num = 0
        for i in range(len(testset)):
            if predicts[i] == dataset.instances[testset[i]][dataset.label_field]:
                correct_num += 1
        return float(correct_num / len(testset))



