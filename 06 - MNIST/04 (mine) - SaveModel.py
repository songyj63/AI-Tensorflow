import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32, [None, 784], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")
# keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool, name="is_training")

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.random_normal([1, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.layers.batch_normalization(L1, training=is_training)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
b2 = tf.Variable(tf.random_normal([1, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.layers.batch_normalization(L2, training=is_training)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b3 = tf.Variable(tf.random_normal([1, 10], stddev=0.01))
model = tf.add(tf.matmul(L2, W3), b3, name="model")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, global_step=global_step)

tf.summary.scalar('cost', cost)

sess = tf.Session()

# global_variables 함수를 통해 앞서 정의하였던 변수들을 저장하거나 불러올 변수들로 설정합니다.
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# 텐서보드에서 표시해주기 위한 텐서들을 수집합니다.
merged = tf.summary.merge_all()
# 저장할 그래프와 텐서값들을 저장할 디렉토리를 설정합니다.
writer = tf.summary.FileWriter('./logs', sess.graph)
# 이렇게 저장한 로그는, 학습 후 다음의 명령어를 이용해 웹서버를 실행시킨 뒤
# tensorboard --logdir=./logs
# 다음 주소와 웹브라우저를 이용해 텐서보드에서 확인할 수 있습니다.
# http://localhost:6006

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(5):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, is_training: True})

        total_cost += cost_val

        summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys, is_training: True})
        writer.add_summary(summary, global_step=sess.run(global_step))

    print('Epoch: %04d' % (epoch + 1), 'Avg. cost = {:.3f}'.format(total_cost / total_batch))

# 최적화가 끝난 뒤, 변수를 저장합니다.
saver.save(sess, './model/dnn.ckpt', global_step=global_step/total_batch)
