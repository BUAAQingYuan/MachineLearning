__author__ = 'PC-LiNing'

import tensorflow as tf
import load_data
import datetime
import argparse
from sklearn.cross_validation import train_test_split

# train
NUM_EPOCHS = 100
BATCH_SIZE = 25
Test_Size = 250
Train_Size = 750
EVAL_FREQUENCY = 10
# svm
svmC = 1
num_features = 2


def train():
    # load data
    data, labels = load_data.extract_data('linearly_separable_data.csv')
    # creating testing and training set
    X_train,X_test,Y_train,Y_test = train_test_split(data,labels,test_size=0.25)

    train_data_node = tf.placeholder(tf.float32, shape=(None, 2))
    train_label_node = tf.placeholder(tf.float32, shape=(None, 1))

    # weight
    W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name="W")
    b = tf.Variable(tf.zeros([1]))
    # y_value = [batch_size,1]
    y_value = tf.matmul(train_data_node, W) + b
    weight_loss = 0.5 * tf.reduce_sum(tf.square(W))
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 1]), 1-train_label_node*y_value))
    svm_loss = weight_loss + svmC * hinge_loss
    # for test
    hinge_loss_test = tf.reduce_sum(tf.maximum(tf.zeros([Test_Size, 1]), 1-train_label_node*y_value))
    svm_loss_test = weight_loss + svmC * hinge_loss_test
    # train
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(svm_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # evaluation
    predicted_class = tf.sign(y_value)
    correct_prediction = tf.equal(train_label_node,predicted_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # runing the training
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print('Initialized!')
        # generate batches
        batches = load_data.batch_iter(list(zip(X_train, Y_train)), BATCH_SIZE, NUM_EPOCHS)
        # batch count
        batch_count = 0
        epoch = 1
        print("Epoch "+str(epoch)+":")
        for batch in batches:
            batch_count += 1
            # train process
            x_batch, y_batch = zip(*batch)
            feed_dict = {train_data_node: x_batch,train_label_node: y_batch}
            _,step,losses = sess.run([train_op, global_step,svm_loss],feed_dict=feed_dict)
            # test process
            if (batch_count * BATCH_SIZE) % Train_Size == 0:
                epoch += 1
                print("Epoch "+str(epoch)+":")
            if batch_count % EVAL_FREQUENCY == 0:
                feed_dict = {train_data_node: X_test,train_label_node: Y_test}
                step,losses,acc = sess.run([global_step,svm_loss_test,accuracy],feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses, acc))


def main(_):
    # if tf.gfile.Exists(FLAGS.summaries_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    # tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='/tmp/svm',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()

