#!/usr/bin/env python3
# -*- coding:UTF-8-*-

__author__ = 'leiser'

'''
Network definition
finished with two parts:network part and check part
'''
import random
from functools import reduce
import numpy as np


class Node(object):
    def __init__(self,layer_index,node_index):
        '''
        # 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，
        实现输出值和误差项的计算。
        节点所属层编号，节点编号
        '''
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.upstream=[]
        self.output=0
        self.delta=0
        
    def set_output(self,output):
        '''设置节点输出值'''
        self.output=output
        
    def append_downstream_connection(self,conn):
        '''添加到下游节点的链接
        conn是一个权重对象(有值，有方法)
        '''
        self.downstream.append(conn)
    
    def append_upstream_connection(self,conn):
        '''添加一个到上游的链接'''
        self.upstream.append(conn)
    
    
    def calc_output(self):
        '''
        根据式1计算节点的输出
        式1：y=sigmoid(w·x)
        '''
        #关于map,reduce：http://blog.csdn.net/damotiansheng/article/details/44139111
        #self.upsstream相当于从上层过来的一个权值和x的列表
        output=reduce(lambda ret,conn: ret+conn.upstream_node.output*conn.weight,self.upstream,0)
        self.output=self.sigmoid(output)
        #print(self.output)检查输出
    
    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，根据式4计算误差delta
        delta(i)=a(i)(1-a(i))sum(w(ki)*delta(k))
        eg:delta(4)=a(4)(1-a(4))sum(w(84)*delta(8)+w(94)*delta(9))
        '''
        downstream_delta=reduce(
            lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,self.downstream,0.0)
        self.delta=self.output*(1-self.output)*downstream_delta
   
    def calc_output_layer_delta(self,label):
        '''
        节点属于输出层时，根据式3计算delta
        delta(i)=y(i)(1-y(i))(t(i)-y(i))
        '''
        self.delta=self.output*(1-self.output)*(label-self.output)
    
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def __str__(self):
        '''打印节点信息'''
        node_str='%u-%u:output:%f delta:%f'%(self.layer_index,self.node_index,self.output,self.delta)
        downstream_str=reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.downstream,'')#'\n\t'是换行的意思
        upstream_str=reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.upstream,'')
        return node_str+'\n\tdownstream:'+downstream_str+'\n\tupstream:'+upstream_str

class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        构造bias这个节点对象，令输出恒为1
        layer_index: 节点所属的层的编号
        node_index: 节点的编号
        '''    
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
    def append_downstream_connection(self, conn):
        '''
        添加一个到下游节点的连接
        '''       
        self.downstream.append(conn)
    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，根据式4计算delta
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta
    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

class Layer(object):
    def __init__(self,layer_index,node_count):
        '''
        Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作。
        层编号，该层的节点数
        '''
        self.layer_index=layer_index
        self.nodes=[]
        #给该层添加节点信息
        for i in range(node_count):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,node_count))#添加常数节点

    def set_output(self,data):
        '''
        设置层的输出，当为输入层时会用到。，这里的data就是calc_output得到的值
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        计算输出向量，
        '''
        for node in self.nodes[:-1]: #除了最后一个元素之外
            node.calc_output()        

    def dump(self):
        '''
        打印节点信息
        '''
        for node in self.nodes:
            print (node)


class Connection(object):
    def __init__(self,upstream_node,downstream_node):
        '''
        Connection对象，主要职责是记录连接的权重，以及这个连接所关联的上下游节点。
        权重初始化为一个很小的随机数
        权重，梯度，更新权重
        '''
        self.upstream_node=upstream_node
        self.downstream_node=downstream_node
        self.weight=random.uniform(-0.1,0.1) #weight初始化随机数
        self.grandient=0.0
        self.cacl_times=0.0

    def calc_gradient(self):
        '''
        计算梯度
        '''
        self.grandient=self.downstream_node.delta*self.upstream_node.output

    def get_gradient(self):
        '''
        获取当前梯度
        '''
        return self.grandient

    def update_weight(self,rate):
        '''
        梯度下降算法更新权重
        '''
        self.calc_gradient()
        self.weight+=rate*self.grandient


    def __str__(self):
        '''
        打印信息
        '''
        return '(%u-%u)->(%u-%u)=%f'%(
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight) 

class Connections(object):
    '''
    Connections对象，提供Connection集合操作。
    '''
    def __init__(self):
        self.connections=[]

    def add_connections(self,connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print (conn)

class Network(object):
    def __init__(self,layers):
        '''
        初始化一个全连接神经网络
        layers：元素位置表示第几层，元素值代表该层节点数
        '''
        self.connections=Connections()
        self.layers=[]
        layer_count=len(layers) #层数
        node_count=0;
        #设置层数和节点数
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i])) #(层数，节点数)
        #设置权重（权重层数比网络层数少1）
        for layer in range(layer_count-1):#双循环定义添加上下层节点，下层节点最后一个不给，因为是b
            connections=[Connection(upstream_node,downstream_node)
                        for upstream_node in self.layers[layer].nodes
                        for downstream_node in self.layers[layer+1].nodes[:-1]]
            #添加权重
            for conn in connections:
                self.connections.add_connections(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self,labels,data_set,rate,iteration):
        '''
        训练神经网络
        labels: 数组，训练样本标签。每个元素是一个样本的标签。
        data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        '''
        #print(labels[0:5])检查；label结构
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)
    
    def train_one_sample(self,label,sample,rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def predict(self,sample):
        '''
        根据输入的样本预测输出值
        sample: 数组，样本的特征，也就是网络的输入向量
        '''
        self.layers[0].set_output(sample)

        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()

        #至此，已经完成6个几点的输出运算，最后return 是为了检查梯度时调用

        return map(lambda node:node.output,self.layers[-1].nodes[:-1]) #layers[-1]是最后一层，nodes[:-1]是除最后一个节点之外的其他节点集合
        #output=[] 
        #for i in range(len(self.layers[1].nodes)):
        #    output.append(self.layers[1].nodes[i].output)
        #return output

    def calc_delta(self,label):
        '''
        内部函数，计算每个节点的delta
        '''
        output_nodes=self.layers[-1].nodes  #最后一层
        #计算输出层delta
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
            #print(output_nodes[i].delta)#检查误差
        
        #print(len(self.layers[1:len(self.layers)-1]))
        
        #计算输隐藏层label
        for layer in self.layers[1:len(self.layers)-1]: #切片操作http://blog.csdn.net/Evan123mg/article/details/49232089
            for node in layer.nodes[:-1]:
                node.calc_hidden_layer_delta()
                #print(node.delta)#检查误差


    def update_weight(self,rate):
        '''
        内部函数，更新每个连接权重
        '''
        for layer in self.layers[:-1]:#除输出层之外的其它层
            for node in layer.nodes:             
                for conn in node.downstream:
                    conn.update_weight(rate)


    def get_gradient(self,label,sample):
        '''
        获得网络在一个样本下，每个连接上的梯度
        label: 样本标签
        sample: 样本输入
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def calc_gradient(self):
        '''
        内部函数，计算每个连接的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def dump(self):
        '''
        打印节点信息
        '''
        for layer in self.layers:
            layer.dump()



#----------------检查网络正确性--------------------

def gradient_check(network,sample_feature,sample_label):
    '''
    梯度检查
    神经网络对象，样本输入，样本label
    '''
    #计算网络误差 1/2sum((y-t)^2)
    network_error=lambda vec1,vec2:\
        0.5*reduce(lambda a,b:a+b,
                map(lambda v:(v[0]-v[1])*(v[0]-v[1]),zip(vec1,vec2)))

    #获取网络在当前样本下的每个链接的梯度
    network.get_gradient(sample_feature,sample_label)

    #对每个梯度做检查
    for conn in network.connections.connections:
        #获取指定连接的梯度
        actucal_gradient=conn.get_gradient()

        #增加一个很小的值，计算网络的误差
        epsilon=0.0001
        conn.weight+=epsilon
        error1=network_error(network.predict(sample_feature),sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)


        #计算出期望梯度，公式见笔记
        expected_gradient=(error2-error1)/(2*epsilon)


        #打印
        print ('expected_gradient:\t%f\nactual_gradient:\t%f'%(
            expected_gradient,actucal_gradient))

def gradient_check_test():
    net=Network([3,2,3])
    sample_feature=[1,1,1]  #输入值得feature和label的变化对结果影响挺大
    sample_label=[1,1,1]
    gradient_check(net,sample_feature,sample_label)

#------------小测一下-----------------

def _test():
    _network=Network([3,5,1])
    _network.train([[1]],[[1,1,1]],0.5,3)  #注意data_set和label的表示[[]]
    _network.dump()

if __name__=='__main__':
    _test()#运行
    #gradient_check_test()#检查网络正确性'


'''
输出结果：

0-0:output:1.000000 delta:-0.000000
        downstream:
        (0-0)->(1-0)=0.009941
        (0-0)->(1-1)=-0.030778
        (0-0)->(1-2)=-0.077085
        (0-0)->(1-3)=0.046720
        (0-0)->(1-4)=-0.047097
        upstream:
0-1:output:1.000000 delta:-0.000000
        downstream:
        (0-1)->(1-0)=-0.092092
        (0-1)->(1-1)=-0.057865
        (0-1)->(1-2)=0.069307
        (0-1)->(1-3)=0.068514
        (0-1)->(1-4)=0.022728
        upstream:
0-2:output:1.000000 delta:-0.000000
        downstream:
        (0-2)->(1-0)=0.064660
        (0-2)->(1-1)=-0.036083
        (0-2)->(1-2)=-0.046965
        (0-2)->(1-3)=-0.043782
        (0-2)->(1-4)=0.080516
        upstream:
0-3: output: 1
        downstream:
        (0-3)->(1-0)=0.067989
        (0-3)->(1-1)=0.041969
        (0-3)->(1-2)=-0.098753
        (0-3)->(1-3)=0.039313
        (0-3)->(1-4)=-0.031305
1-0:output:0.513135 delta:-0.001028
        downstream:
        (1-0)->(2-0)=0.001187
        upstream:
        (0-0)->(1-0)=0.009941
        (0-1)->(1-0)=-0.092092
        (0-2)->(1-0)=0.064660
        (0-3)->(1-0)=0.067989
1-1:output:0.478123 delta:0.002404
        downstream:
        (1-1)->(2-0)=0.105556
        upstream:
        (0-0)->(1-1)=-0.030778
        (0-1)->(1-1)=-0.057865
        (0-2)->(1-1)=-0.036083
        (0-3)->(1-1)=0.041969
1-2:output:0.461246 delta:0.000915
        downstream:
        (1-2)->(2-0)=0.058286
        upstream:
        (0-0)->(1-2)=-0.077085
        (0-1)->(1-2)=0.069307
        (0-2)->(1-2)=-0.046965
        (0-3)->(1-2)=-0.098753
1-3:output:0.528221 delta:-0.001119
        downstream:
        (1-3)->(2-0)=-0.000756
        upstream:
        (0-0)->(1-3)=0.046720
        (0-1)->(1-3)=0.068514
        (0-2)->(1-3)=-0.043782
        (0-3)->(1-3)=0.039313
1-4:output:0.505998 delta:0.000425
        downstream:
        (1-4)->(2-0)=0.045809
        upstream:
        (0-0)->(1-4)=-0.047097
        (0-1)->(1-4)=0.022728
        (0-2)->(1-4)=0.080516
        (0-3)->(1-4)=-0.031305
1-5: output: 1
        downstream:
        (1-5)->(2-0)=-0.021710
2-0:output:0.483664 delta:0.128946
        downstream:
        upstream:
        (1-0)->(2-0)=0.001187
        (1-1)->(2-0)=0.105556
        (1-2)->(2-0)=0.058286
        (1-3)->(2-0)=-0.000756
        (1-4)->(2-0)=0.045809
        (1-5)->(2-0)=-0.021710
2-1: output: 1
        downstream:
        
        '''
