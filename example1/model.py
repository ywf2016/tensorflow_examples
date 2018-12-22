import tensorflow as tf
from download import mnist_data as mnist 
from accuracy import accuracy as accuracy_test
from config import LEARNING_RATE

def model(x,y,keep_prob,device = 'gpu'):
    with tf.device('/%s:0' % device):
        with tf.name_scope(device):
            # 输入层
            x_image = tf.reshape(x,[-1,28,28,1])
            # 一个卷积层
            W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
            b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
            h_conv1 = tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding="SAME")
            h_relu1 = tf.nn.relu(h_conv1+b_conv1)
            h_pool1 = tf.nn.max_pool(h_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            # 第二个卷积层
            W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
            b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
            h_conv2 = tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding="SAME")
            h_relu2 = tf.nn.relu(h_conv2+b_conv2)
            h_pool2 = tf.nn.max_pool(h_relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            # 第三个全连接层
            W_fc3 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
            b_fc3 = tf.Variable(tf.constant(0.1,shape=[1024]))
            h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
            h_fc3 = tf.matmul(h_pool2_flat,W_fc3) + b_fc3
            h_relu3 = tf.nn.relu(h_fc3)
            # 第四个dropout
            h_fc4_drop = tf.nn.dropout(h_relu3,keep_prob)
            # 输出层
            W_output = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
            b_output = tf.Variable(tf.constant(0.1,shape=[10]))
            predict = tf.matmul(h_fc4_drop,W_output) + b_output

            # 损失函数交叉熵
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predict))
            # 优化函数
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    return train_step,predict


