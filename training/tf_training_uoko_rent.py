import pandas as pd
import numpy as np
import tensorflow as tf

train_data = pd.read_csv('D://uoko//normalization//normalization_train.csv', header=0, low_memory=False,
                         encoding='utf-8')
test_data = pd.read_csv('D://uoko//normalization//normalization_test.csv', header=0, low_memory=False, encoding='utf-8')
# 训练样本
train_x = train_data.iloc[:, 1:]
train_y = train_data.iloc[:, 0:1]
print(train_x.shape[1])
# 测试样本
x_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0:1]


def add_layer(inputs, input_size, output_size, activation_function=None):
    with tf.variable_scope("Weights"):
        Weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name="weights")
    with tf.variable_scope("biases"):
        biases = tf.Variable(tf.zeros(shape=[1, output_size]) + 0.1, name="biases")
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
    with tf.name_scope("dropout"):
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=keep_prob_s)
    if activation_function is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_function"):
            return activation_function(Wx_plus_b)


xs = tf.placeholder(shape=[None, train_x.shape[1]], dtype=tf.float32, name="inputs")
ys = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y_true")
keep_prob_s = tf.placeholder(dtype=tf.float32)
# 第一层
with tf.name_scope("layer_1"):
    L1 = add_layer(xs, 45, 60, activation_function=tf.nn.relu)
# 第二层
with tf.name_scope("layer_2"):
    L2 = add_layer(L1, 60, 60, activation_function=tf.nn.relu)
# 第三层
with tf.name_scope("layer_2"):
    L3 = add_layer(L2, 60, 60, activation_function=tf.nn.relu)

# 输出层
with tf.name_scope("y_pred"):
    pred = add_layer(L3, 60, 1, activation_function=tf.nn.relu)

# 损失函数
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred), reduction_indices=[1]))  # mse
    tf.summary.scalar("loss", tensor=loss)
# 梯度下降
# lr = tf.Variable(0.1, dtype=tf.float32)
with tf.name_scope("train"):
    # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()
keep_prob = 1
ITER = 500
# 训练
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(ITER):
        # sess.run(tf.assign(lr, 0.1 * (0.95 ** epoch)))
        train_acc, _ = sess.run([loss, train_op], feed_dict={xs: train_x, ys: train_y, keep_prob_s: keep_prob})
        test_acc = sess.run(loss, feed_dict={xs: x_test, ys: y_test, keep_prob_s: keep_prob})
        if epoch % 10 == 0:
            print("epoch:" + str(epoch) + '  test_acc:' + str(np.sqrt(test_acc)))
