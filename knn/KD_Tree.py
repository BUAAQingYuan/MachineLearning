#coding=utf-8

__author__ = 'PC-LiNing'

import numpy
from collections import namedtuple
import math


class KdNode(object):
    def __init__(self, dom_elt,dom_label,split,left,right):
        self.dom_elt = dom_elt
        self.dom_label = dom_label
        self.split = split
        self.left = left
        self.right = right
        # 节点掩码，当为True时，正常更新;当为False时,不更新。
        self.flag = True


def select_dim(X,dims_num):
    if not X.tolist():
        return 0
    var_dims = numpy.var(X,axis=0).tolist()
    max_var = max(var_dims)
    return var_dims.index(max_var)


class KdTree(object):
    def __init__(self, data,labels):
        k = len(data[0])

        def CreateNode(split, data_set,label_set):
            if not data_set:
                return None
            zipped_data = zip(data_set,label_set)
            all_data = list(zipped_data)
            all_data.sort(key=lambda x: x[0][split])
            split_pos = len(all_data) // 2
            # median = (x,y)
            median = all_data[split_pos]
            # split
            data_left = all_data[:split_pos]
            _left = list(zip(*data_left))
            x_left = list(_left[0]) if len(_left) > 0 else []
            y_left = list(_left[1]) if len(_left) > 0 else []
            data_right = all_data[split_pos + 1:]
            right_ = list(zip(*data_right))
            x_right = list(right_[0]) if len(right_) > 0 else []
            y_right = list(right_[1]) if len(right_) > 0 else []
            # split_next = (split + 1) % k
            split_left = select_dim(numpy.asarray(x_left,dtype=numpy.float32), dims_num=k)
            split_right = select_dim(numpy.asarray(x_right,dtype=numpy.float32), dims_num=k)
            return KdNode(median[0],median[1],split,
                          CreateNode(split_left, x_left,y_left),
                          CreateNode(split_right, x_right,y_right))

        split = select_dim(numpy.asarray(data,dtype=numpy.float32), dims_num=k)
        self.root = CreateNode(split, data,labels)


def preorder(root):
    print(str(root.dom_elt)+" "+str(root.dom_label))
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


def find_nearest(tree,point):
    k = len(point)
    result = namedtuple("Result_tuple", "nearest_point  nearest_node  nearest_dist  nodes_visited")

    def travel(kd_node,target,max_dist):
        if kd_node is None :
            return result([0] * k, None, float("inf"), 0)
        nodes_visited = 1
        s = kd_node.split
        pivot = kd_node.dom_elt
        if target[s] <= pivot[s]:
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left
        # max_dist is useless
        temp1 = travel(nearer_node,target,max_dist)
        nearest = temp1.nearest_point
        dist = temp1.nearest_dist
        nearest_node = temp1.nearest_node
        nodes_visited += temp1.nodes_visited
        if dist < max_dist and nearest_node.flag == True:
            # min dist
            max_dist = dist
        # joint distance at dim
        # 如果某一维相交距离都大于当前最小距离，那么真实距离肯定更大
        temp_dist = abs(pivot[s] - target[s])
        if max_dist < temp_dist:
            # no joint
            return result(nearest,nearest_node,dist,nodes_visited)
        # 如果相交，再比较真实距离
        # point distance
        temp_dist = math.sqrt(sum((p1 -p2)**2 for p1,p2 in zip(pivot,target)))
        if temp_dist < dist and kd_node.flag is True:
            nearest = pivot
            nearest_node = kd_node
            dist = temp_dist
            max_dist = dist
        # 向上回退过程中，如果父节点相交，那么要检查另一个子节点，如果不想交，不必检查另一个子节点
        temp2 = travel(further_node,target,max_dist)
        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            nearest_node = temp2.nearest_node
            dist = temp2.nearest_dist
        return result(nearest,nearest_node,dist,nodes_visited)

    return travel(tree.root,point,float("inf"))


def find_k_nearest(tree,point,num):
    k_nodes = []
    for i in range(num):
        i_nearest = find_nearest(tree,point)
        # set flag = False
        i_nearest.nearest_node.flag = False
        k_nodes.append(i_nearest)
    # recovery
    for i_near in k_nodes:
        i_near.nearest_node.flag = True
    return k_nodes
