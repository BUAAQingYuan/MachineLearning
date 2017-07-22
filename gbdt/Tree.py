__author__ = 'PC-LiNing'


class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        # real特征 <= split_value,放到左子树
        # 非real特征 = split_value,放到左子树
        self.split_value = None
        self.is_real_feature = True
        # 每棵树只关心叶节点
        self.leafNode = None

    def get_predict_value(self, instance):
        # 到达叶节点
        if self.leafNode:
            return self.leafNode.get_weight()
        if self.is_real_feature and instance[self.split_feature] <= self.split_value:
            return self.leftTree.get_predict_value(instance)
        elif not self.is_real_feature and instance[self.split_feature] == self.split_value:
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)


class LeafNode:
    def __init__(self,idset):
        self.idset = idset
        self.weight = None

    def get_idset(self):
        return self.idset

    def get_weight(self):
        return self.weight

    def update_weight(self, next_gradients):
        sum1 = sum([next_gradients[id] for id in self.idset])
        if sum1 == 0:
            self.weight = sum1
            return
        sum2 = sum([abs(next_gradients[id])*(2-abs(next_gradients[id])) for id in self.idset])
        self.weight = sum1 / sum2


def compute_mean_square(values):
    if len(values) < 2:
        return 0
    mean = sum(values) / float(len(values))
    error = 0.0
    for value in values:
        error += (mean - value) ** 2
    return error


def build_decision_tree(dataset,subset,next_gradient,current_depth,leafnodes,max_depth,loss):
    if current_depth < max_depth:
        # sample 多少种特征进行训练
        features = dataset.get_features()
        mse = -1
        selectedFeature = None
        split_value = None
        select_left_ids = []
        select_right_ids = []
        for feature in features:
            feature_values = dataset.get_all_values(feature)
            is_real_type = dataset.is_distinct_type(feature)
            for value in feature_values:
                left_ids = []
                right_ids = []
                for id in subset:
                    item = dataset.instances[id]
                    id_value = item[feature]
                    # 分裂节点
                    if (is_real_type and id_value <= value) or (not is_real_type and id_value == value):
                        left_ids.append(id)
                    else:
                        right_ids.append(id)
                # 由负梯度拟合构建树
                left_gradients = [next_gradient[id] for id in left_ids]
                right_gradients = [next_gradient[id] for id in right_ids]
                sum_mse = compute_mean_square(left_gradients) + compute_mean_square(right_gradients)
                if mse < 0 or sum_mse < mse:
                    select_left_ids = left_ids
                    select_right_ids = right_ids
                    mse = sum_mse
                    selectedFeature = feature
                    split_value = value
        if not selectedFeature or mse < 0:
            raise ValueError("cannot determine the split attribute.")
        tree = Tree()
        tree.split_feature = selectedFeature
        tree.is_real_feature = dataset.is_distinct_type(selectedFeature)
        tree.split_value = split_value
        tree.leftTree = build_decision_tree(dataset,select_left_ids,next_gradient,current_depth+1,leafnodes,max_depth,loss)
        tree.rightTree = build_decision_tree(dataset,select_right_ids,next_gradient,current_depth+1,leafnodes,max_depth,loss)
        return tree
    else:
        # 已经达到最大深度，不需要在构建树
        node = LeafNode(subset)
        node.update_weight(next_gradient)
        leafnodes.append(node)
        tree = Tree()
        tree.leafNode = node
        return tree



