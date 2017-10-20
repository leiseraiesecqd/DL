# python 3.6

import gzip, os
from struct import unpack
import numpy as np
from array import array


def load_mnist(dataset='_', path='.'):
    """
    loosely based on
    https://github.com/amitgroup/amitgroup/blob/master/amitgroup/io/mnist.py
    https://gist.github.com/akesling/5358964
    """
    if dataset == "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte.gz')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    elif dataset == "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte.gz')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    else:
        raise ValueError("dataset must be 'test' or 'train'")

    with gzip.open(fname_img, 'rb') as f:
        #解析头文件参数依次为魔数、图片数量、每张图片高、每张图片宽
        mn, ni, nr, nc = unpack('>IIII', f.read(16))#前16位是头信息
        img_data = array('B', f.read())#读取
        img_data = np.array(img_data, dtype=np.uint8)#unit8（无符号的整数，unit8是0～255）
        img_data = img_data.reshape(ni, nr, nc)#重构

    with gzip.open(fname_lbl, 'rb') as f:
        #解析参数依次为魔数、图片数量
        mn, nl = unpack('>II', f.read(8))
        lbl_data = array('B', f.read())
        lbl_data = np.array(lbl_data, dtype=np.uint8)

    return img_data, lbl_data

'''
if __name__ == '__main__':
    Xtrain, Ytrain = load_mnist('train', '')
    Xtest, Ytest = load_mnist('test', '')
    print(type(Xtrain))
    print(Xtest.shape)
    #print(Xtr[0])
    #print(len(Ytr))

'''