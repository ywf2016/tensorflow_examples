import tensorflow as tf
'''
如果一次把测试数据全部放进去会导致OOM，这里分组进行测试，最后统计结果
'''
# Session要用原来的Session，如果更换新的Session会导致原本权值丢失
BATCH_SIZE = 100
def accuracy(sess,predict,test_data,x,y,keep_prob):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(predict,1))
    correct_count = 0
    all_count = 0
    for _ in range(100): 
        images,labels = test_data.next_batch(BATCH_SIZE)
        correct = sess.run(correct_prediction,feed_dict={x:images,y:labels,keep_prob:1.0})
        correct_count = correct_count + sum(correct)
        all_count = all_count + len(correct)
    return correct_count / all_count