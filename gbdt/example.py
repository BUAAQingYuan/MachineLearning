__author__ = 'PC-LiNing'

from gbdt import model
from gbdt import Data
from random import sample


if __name__ == '__main__':
    data_file = './gbdt/credit.data.csv'
    dateset = Data.Dataset(data_file)
    gbdt = model.GBDT(max_iter=20, sample_rate=0.8, learning_rate=0.5, max_depth=7, loss_type='binary-classification')
    train_sample_rate = 0.75
    train_ids = sample(list(range(dateset.size)), int(dateset.size * train_sample_rate))
    test_ids = list(set(range(dateset.size)) - set(train_ids))
    # train
    gbdt.train(dateset, train_ids)
    # test
    acc = gbdt.compute_acc(dateset, test_ids)
    print("acc: "+str(acc))
