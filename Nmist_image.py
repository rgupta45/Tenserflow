# Classification of hand written digits Nmist

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros(10))

sess.run(tf.global_variables_initializer())

pred_out = tf.matmul(X,W)+ b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred_out)
)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={X: batch[0],y:batch[1]})

correct_pred = tf.equal(tf.argmax(pred_out,1),tf.argmax(y,1))
accuracy =tf.reduce_mean(tf.cast(correct_pred,tf.float32))
print(accuracy.eval(feed_dict={X: mnist.test.images,y: mnist.test.labels}))


