
### KNN


#### KD-tree

KD-tree 只能用于欧式距离，并且处理高维数据效果不佳(D<=20)。

训练数据量 N >> 2^D。

对于两个数据点 x,y

|| x - y || >= || x_i - y_i || 

#### Error analyse

最近邻(k=1)的错误率高于贝叶斯(贝叶斯最优分类器)错误率，但错误率不超过贝叶斯最优分类器的两倍。

P* <= P <= 2P*

对于KNN，当样本数据量充足时，k-近邻法的错误率要低于最近邻法。

![knn error](https://github.com/BUAAQingYuan/MachineLearning/raw/master/knn/knn_error.jpg)


#### Reference

[SciPy KDTree](http://scipy-cookbook.readthedocs.io/items/KDTree_example.html)

[KNN算法与Kd树](http://www.cnblogs.com/21207-iHome/p/6084670.html)
