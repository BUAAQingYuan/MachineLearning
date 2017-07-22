#coding=utf-8

__author__ = 'PC-LiNing'

import numpy as np
import pandas as pd
from pandas import DataFrame


def load_data(csv_file):
    df = pd.read_csv("credit.data.csv",header=0)
    df.columns = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','label']
    X = df[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']]
    X = np.array(X)
    Y = df["label"]
    Y = np.array(Y)
    return X, Y


class Dataset(object):
    def __init__(self,csv_file, label='label'):
        line_count = 0
        self.distinct_valueset = dict()
        self.instances = dict()
        self.label_field = label
        for line in open(csv_file):
            if line == "\n":
                continue
            fields = line[:-1].split(",")
            if line_count == 0:
                self.field_names = tuple(fields)
            else:
                # 判断特征类型
                if line_count == 1:
                    self.field_type = dict()
                    for i in range(len(self.field_names)):
                        valueSet = set()
                        try:
                            # 连续值
                            float(fields[i])
                            # 存储连续值特征的取值
                            self.distinct_valueset[self.field_names[i]] = set()
                        except ValueError:
                            # 离散值
                            valueSet.add(fields[i])
                        # 存储离散值特征的取值
                        # 若field_type[field_name]的长度为0，则为连续值
                        self.field_type[self.field_names[i]] = valueSet
                self.instances[line_count-1] = self.construct_instance(fields)
            line_count += 1
        self.size = line_count - 1

    def construct_instance(self,fields):
        instance = dict()
        for i in range(len(fields)):
            field_name = self.field_names[i]
            is_distinct = self.is_distinct_type(field_name)
            if is_distinct:
                instance[field_name] = float(fields[i])
                self.distinct_valueset[self.field_names[i]].add(float(fields[i]))
            else:
                instance[field_name] = fields[i]
                self.field_type[self.field_names[i]].add(fields[i])
        return instance

    def is_distinct_type(self,field_name):
        return len(self.field_type[field_name]) == 0

    def describe(self):
        print("features:"+str(self.field_names))
        print("dataset size="+str(self.size))
        for field in self.field_names:
            if self.is_distinct_type(field):
                print(field+" "+"float")
                print(str(self.distinct_valueset[field]))
            else:
                print(field+" "+"class")
                print(str(self.field_type[field]))

    def get_label_size(self):
        name = self.label_field
        return len(self.field_type[name]) or len(self.distinct_valueset[name])

    def get_features(self):
        features = [x for x in self.field_names if x != self.label_field]
        return tuple(features)

    def get_all_values(self,field_name):
        if self.is_distinct_type(field_name):
            return self.distinct_valueset[field_name]
        else:
            return self.field_type[field_name]

"""
csv_file = 'credit.data.csv'
dataset = Dataset(csv_file)
dataset.describe()
print(dataset.instances[0])
"""

