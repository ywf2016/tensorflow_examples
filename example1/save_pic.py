# 将数据集保存维图片
from download import mnist_data as mnist
import scipy.misc
import os
import numpy as np
# 创建文件夹
save_dir = 'MNIST.data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存图片
for i in range(20):
    image_array = mnist.train.images[i,:]
    image_array = image_array.reshape(28,28)
    filename = save_dir + str(i) +".jpg"
    scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)
    one_hot_label = mnist.train.labels[i,:]
    label = np.argmax(one_hot_label)
    print(str(i)+','+str(label))
