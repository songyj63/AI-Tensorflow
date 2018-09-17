import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


saver = tf.train.import_meta_graph("./model/dnn.ckpt-15.meta")
sess = tf.Session()
saver.restore(sess, "./model/dnn.ckpt-15")

model = sess.graph.get_tensor_by_name('model:0')
X = sess.graph.get_tensor_by_name('X:0')
Y = sess.graph.get_tensor_by_name('Y:0')
is_training = sess.graph.get_tensor_by_name('is_training:0')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))*100

print('정확도: %.2f %%' %sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, is_training: False}))

labels = sess.run(model, feed_dict={X: mnist.test.images,
                                    Y: mnist.test.labels,
                                    is_training: False})

fig = plt.figure()

for i in range(10):
    subplot = fig.add_subplot(2, 5, i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)

plt.show()