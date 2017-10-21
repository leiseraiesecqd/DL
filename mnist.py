#!/usr/bin/env python3
# -*- coding:UTF-8-*-

__author__ = 'leiser'

'''
data_set processing
'''
import struct
import numpy as np
from network import Network
from read import load_mnist
from datetime import datetime

#-------------数据预处理-----------------

#将label进行one-hot encoding
def norm(label):
	'''
    #将一个值转换为10维标签向量
	'''
	#建立二维数组
	label_vec=[[] for d in range(len(label))]
	# print('label is \n')
    # print(label[:20])

	for i in range(len(label)):
		for j in range(10):
			if j==label[i]:
				label_vec[i].append(1)
			else:
				label_vec[i].append(0)
	#将list变成np.array
	return np.array(label_vec)


#调用read函数读取数据集
train_data_set,train_labels = load_mnist('train', '')
print('train_set is ready'+str(type(train_data_set)))
train_labels=norm(train_labels)
print('train_labels finised one-hot encoding'+str(type(train_labels)))
test_data_set,test_labels = load_mnist('test', '')
print('test_set is ready')
test_labels=norm(test_labels)
print('test_labels finised one-hot encoding')


#内存不够用，先跑几张
train_data_set=train_data_set[0:10].reshape(10,784)#一张图片train_data_set[0].reshape(1,784)
train_labels=train_labels[0:10].reshape(10,10,1)#一张图片train_labels[0].reshap(10,1)
test_data_set=test_data_set[0:5].reshape(5,784)
test_labels=test_labels[0:5].reshape(5,10,1)

#----------检查数据格式-------------
#print(train_data_set)
#print(train_labels)
#print('\n')
#print(test_labels)
'''
print('#########################')
print(train_data_set.shape)
print(train_labels.shape)
print(test_data_set.shape)
print(test_labels.shape)
print('#########################')
print(train_data_set)
print('\n\n\n\n')
print(train_labels)
print('\n\n\n\n')
print(test_data_set)
print('\n\n\n\n')
print(test_labels)
print('#########################')
'''
#---------框架主体----------------

#获取label中真实值
def get_result(vec):
	'''
	获取预测结果
	'''
	max_value_index=0
	max_value=0
	for i in range(len(vec)):
		if vec[i]>max_value:
			max_value=vec[i]
			max_value_index=i
	return max_value_index


def evaluate(network,test_data_set,test_labels):
	'''
	评估网络error
	'''
	error=0
	total=len(test_data_set)
	#print(total)
	#print(type(test_labels))
	for i in range(total):
		label=get_result(test_labels[i])#label的de-encoding，变成数字
		predict=get_result(network.predict(test_data_set[i]))
		#print(network.predict(test_data_set[i]))检查预测结果
		#print(label)
		print(predict)
		if label !=predict:
			error+=1
	return float(error)/float(total)

def train_and_evaluate():
	last_error_ratio=1.0
	epoch=0
	#print ('[dataset label:]\n')
	#print (train_labels[:10])
	network=Network([784,300,10])
	print('now start!!!')
	while True:
		epoch+=1
		network.train(train_labels,train_data_set,0.3,1)
		print ('%s epoch %d finished'%(datetime.now(),epoch))
		if epoch %1==0:
			error_ratio=evaluate(network,test_data_set,test_labels)
			print('%s after epoch %d,error ratio is %f'%(datetime.now(),epoch,error_ratio))
			#当准确率开始下降时（出现了过拟合）,终止训练。
			if error_ratio>last_error_ratio:
				break
			else:
				last_error_ratio=error_ratio


#-------------------主函数------------------
if __name__=='__main__':
	#print(get_result(train_labels))检查获得标签值函数是否正确
	#print(get_result(test_labels))
	train_and_evaluate()