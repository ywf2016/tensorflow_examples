import tensorflow as tf
from time import time
from download import mnist_data as mnist 
from accuracy import accuracy as accuracy_test
from config import BATCH_SIZE,TRAIN_STEPS,LEARNING_RATE,KEEP_PROB
from model import model

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

train_step,predict = model(x,y,keep_prob,'gpu')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
starttime = time()
for i in range(TRAIN_STEPS):
    images,labels = mnist.train.next_batch(BATCH_SIZE)
    train_step.run(feed_dict={x:images,y:labels,keep_prob:KEEP_PROB})
endtime = time()
print("训练步数：%d，总共耗时%fs" % (TRAIN_STEPS,endtime-starttime))
print("测试集的准确率为:%f" % accuracy_test(sess,predict,mnist.test,x,y,keep_prob))