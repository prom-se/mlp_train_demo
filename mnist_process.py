import numpy as np
import struct

from PIL import Image
import os

dataset_path = 'data/MNIST/raw/'
output_path = 'data/MNIST/processed/'
train_image_file = dataset_path + 'train-images-idx3-ubyte'
train_label_file = dataset_path + 'train-labels-idx1-ubyte'

test_image_file = dataset_path + 't10k-images-idx3-ubyte'
test_label_file = dataset_path + 't10k-labels-idx1-ubyte'

train_path = output_path + 'mnist_train'
test_path = output_path + 'mnist_test'

train_size = 47040016
test_size = 7840016

train_label_size = 60008
test_label_size = 10008

train_size = str(train_size - 16) + 'B'
test_size = str(test_size - 16) + 'B'
train_label_size = str(train_label_size - 8) + 'B'
test_label_size = str(test_label_size - 8) + 'B'


def process(img, size, label, lable_size, path):
    data_buf = open(img, 'rb').read()

    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)
    datas = struct.unpack_from('>' + size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)

    label_buf = open(label, 'rb').read()
    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from('>' + lable_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(10):
        file_name = path + os.sep + str(i)
        if not os.path.exists(file_name):
            os.mkdir(file_name)

    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = path + os.sep + str(label) + os.sep + str(ii) + '.png'
        print(file_name)
        img.save(file_name)


if __name__ == '__main__':
    process(train_image_file, train_size, train_label_file, train_label_size, train_path)
    process(test_image_file, test_size, test_label_file, test_label_size, test_path)
