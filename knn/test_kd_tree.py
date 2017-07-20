__author__ = 'PC-LiNing'

from knn import KD_Tree

data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
labels = [10,20,30,40,50,60]
kd = KD_Tree.KdTree(data,labels)
# KD_Tree.preorder(kd.root)
point = [3,4.5]
print('Find nearest neighbor:')
nearst = KD_Tree.find_nearest(kd,point)
print(nearst)
print(nearst.nearest_node.dom_elt)

print('Find k nearest neighbor:')
k_nearest = KD_Tree.find_k_nearest(kd,point,num=3)
for i_nearest in k_nearest:
    print(str(i_nearest.nearest_node.dom_elt) + " "+str(i_nearest.nearest_node.dom_label))

print('Find nearest neighbor,test the K-D tree is recovery:')
nearst2 = KD_Tree.find_nearest(kd,point)
print(nearst2)
print(nearst2.nearest_node.dom_elt)