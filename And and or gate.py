# Single perceptron and gate implementation in tenserflow
import tensorflow as tf
T,F =1.,-1.
bias =1.
train_in = [
    [T,T,bias],
    [T,F,bias],
    [F,T,bias],
    [F,F,bias],
]
train_out =[
    [T],
    [F],
    [F],
    [F],
]
W = tf.Variable(tf.random_normal([3,1]))
init = tf.initialize_all_variables()
sess= tf.Session()
sess.run(init)
# Creating a step function:
def step(x):
    is_greator = tf.greater(x,0)
    flot_greator = tf.to_float(is_greator)
    double = tf.multiply(flot_greator,2)
    return  tf.subtract(double,1)
output = step((tf.matmul(train_in,W)))
error = tf.subtract(train_out,output)
mse = tf.reduce_mean(tf.square(error))
# Weight reasignment
delta = tf.matmul(train_in,error,transpose_a=True)
train = tf.assign(W,tf.add(W,delta))
err,  target =  1,0
epoch, max_epoch= 0 , 10
while err >target and epoch < max_epoch:
    epoch += 1
    err, _ = sess.run([mse,train])
    print(err, epoch)
