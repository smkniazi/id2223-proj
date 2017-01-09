
# coding: utf-8

# In[2]:

import os
import tensorflow as tf
import numpy as np
import math, random
import pylab as pl
from IPython import display

print "Starting ... "

########################################
# START Reading the data set 
########################################
data_path="/root/data/"

training_data_size=360000
train_start_index=0
train_end_index=(training_data_size-1)
train_index=train_start_index;
test_start_index=train_end_index+1
test_end_index=(400000-1)
test_index=test_start_index;

def nextTrainingBatch(batch_size):
    global train_index
    data_arr, label_arr, file_names, counter =  readData(data_path, batch_size, train_index)
    train_index += counter
    if train_index >= train_end_index: 
        train_index = train_start_index 
        print "Warning: Finished reading the entire training dataset. Next training data batch request will reuse the samples"
    return data_arr, label_arr, file_names

def nextTestingBatch(batch_size):
    global test_index
    data_arr, label_arr, file_names, counter =  readData(data_path, batch_size, test_index)
    test_index += counter
    if test_index >= test_end_index:
        test_index = test_start_index
        print "Warning: Finished reading the entire test dataset. Next test data batch request will reuse the samples"
    return data_arr, label_arr, file_names

minT=-80.0
maxT=29.988
meanT = -3.89422067546
stdT = 33.3924274674
    
minQ=0.0
maxQ=27.285
meanQ = 13.1465901805
stdQ = 9.33156561388

minR=-9.94
maxR=7.455
meanR = -2.07510301805
stdR = 1.21609343765

def normalizeT(t):
    return normalize(t, minT, maxT, meanT, stdT)

def normalizeQ(q):
    return normalize(q, minQ, maxQ, meanQ, stdQ)

def normalizeR(r):
    return normalize(r, minR, maxR, meanR, stdR)

def normalize(x, min, max, mean, std):
    #return  (x - min) / (max - min) # min max normalization
    return (x - mean)/std # standardization  
    #return x+100

def readData(data_path, batch_size, index):
    data_arr=[]
    label_arr=[]
    file_names_arr=[]
    counter=0
    for i in range(0, batch_size):
        fileToRead = os.path.join(data_path,str(index)+".csv")
        if os.access(fileToRead, os.R_OK):
            f = open(fileToRead)
            index+=1
            counter+=1
            lines = f.readlines()
            data=[]
            label=[]
            for j in range(1,len(lines)):
                items = lines[j].strip().split(",")
                data.append(normalizeT(float(items[2])))
                data.append(normalizeQ(float(items[3])))
                label.append(normalizeR(float(items[4])))
            f.close()
            
            for p in range (0, 12):
                data.append(0.0)
                
            data_arr.append(data)
            label_arr.append(label)  
            file_names_arr.append(index)
        else: 
            print "Unable to read the file "+fileToRead
    return data_arr, label_arr, file_names_arr, counter
    makePredictions(data_arr, label_arr, "Training Sample ")

#if __name__ == "__main__":
#   data_arr, label_arr = nextTrainingBatch(2)
#   print data_arr
#   
#   data_arr, label_arr = nextTestingBatch(2 )
#   print label_arr

########################################
# END Reading the data set 
########################################

c1_size=32
c2_size=64
c3_size=128
fc_size1 = 512
fc_size2 = 256
weight_stddev=0.3
bias_stddev=0.03

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

def pool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def ReLU(x):
    #return tf.nn.relu(x)
    return leakyReLU(x,0.001)

def Sgimod(x):
    return tf.nn.sigmoid(x)

def weightInitilization5(a,b,c,d, wstddev):
    return tf.Variable(tf.random_normal([a, b, c, d], stddev=wstddev))

def weightInitilization3(a,b, wstddev):
    return tf.Variable(tf.random_normal([a, b], stddev=wstddev))

# in the lecture 5 slide 38 set b to small value i.e. 0.1
def biasInitialization(a,bstddev):
    return tf.Variable(tf.random_normal([a],stddev=bias_stddev, mean=0.1))
    #return tf.Variable(tf.zeros([a]))
    
#https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/V6aeBw4nlaE
def leakyReLU(x, alpha=0., max_value=None):
    '''Rectified linear unit
    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    if alpha != 0.:
        negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        max_value = _to_tensor(max_value, x.dtype.base_dtype)
        zero = _to_tensor(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)
    if alpha != 0.:
        x -= alpha * negative_part
    return x



weights = {
    'wc1': weightInitilization5(2, 2, 1, c1_size, weight_stddev),
    'wc2': weightInitilization5(2, 2, c1_size, c2_size, weight_stddev),
    'wc3': weightInitilization5(2, 2, c2_size, c3_size, weight_stddev),
    'wf1': weightInitilization3(2*2*c3_size, fc_size1, weight_stddev),
    'wf2': weightInitilization3(fc_size1, fc_size2, weight_stddev),
    'out': weightInitilization3(fc_size2, 26, weight_stddev)
}

biases = {
    'bc1': biasInitialization(c1_size, bias_stddev),
    'bc2': biasInitialization(c2_size, bias_stddev),
    'bc3': biasInitialization(c3_size, bias_stddev),
    'bf1': biasInitialization(fc_size1, bias_stddev),
    'bf2': biasInitialization(fc_size2, bias_stddev),
    'out': biasInitialization(26, bias_stddev)
}

# Create model
def conv_net(x, weights, biases, dropout):
    # x is 64 x 1 tensor with padding at the end
    x = tf.reshape(x, shape=[-1, 8, 8, 1])
    
    # first convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides=1)
    conv1 = ReLU(conv1)
    conv1 = pool2d(conv1, k=1)
    
    # second convolution layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides=1)
    conv2 = ReLU(conv2)
    conv2 = pool2d(conv2, k=2)
    
    # third convolution layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides=1)
    conv3 = ReLU(conv3)
    conv3 = pool2d(conv3, k=2)
    
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wf1'].get_shape().as_list()[0]])
    
    # Fully connected layer 1
    fc1 = tf.add(tf.matmul(fc1, weights['wf1']), biases['bf1'])
    fc1 = ReLU(fc1)
    #fc1 = tf.nn.dropout(fc1, dropout)
    
    # Fully connected layer 2
    fc2 = tf.add(tf.matmul(fc1, weights['wf2']), biases['bf2'])
    fc2 = ReLU(fc2)
    #fc2 = tf.nn.dropout(fc2, dropout)
    
    # Output radiation prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


# Parameters
NUM_EPOCS=120000
#NUM_EPOCS=30000
BATCH_SIZE=3
TEST_AFTER=100
learning_rate = 0.001
dropout = 1 # Dropout, probability to keep units

n_input = 64
n_output = 26
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

#construct model
pred = conv_net(x, weights, biases, keep_prob)

# loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred,y))
#cost = tf.reduce_sum(tf.squared_difference(pred,y)/2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def modelTestError():
    data_arr, label_arr, file_names = nextTestingBatch(BATCH_SIZE)
    mse = sess.run((cost), feed_dict={x: data_arr,  y: label_arr, keep_prob: 1.0})
    return mse

mse_train_graph = []
mse_test_graph = []
mse_xaxis = []
def testErrorAndReplot(epoch, mse_train):
    display.clear_output(wait=True)

    mse_test = modelTestError() 
    
    mse_test_graph.append(mse_test)
    mse_train_graph.append(mse_train)
    mse_xaxis.append(epoch)
    
    zoomLength = 10
    zoomTest = mse_test_graph[-zoomLength:]
    zoomTrain = mse_train_graph[-zoomLength:]
    zoomXasis = mse_xaxis[-zoomLength:]

    pl.figure(figsize=(12, 5))
    pl.subplot(121)
    pl.cla()
    pl.title("Zoomed (Last "+str(zoomLength)+" Epocs)")
    pl.xlabel("Epoc")
    pl.ylabel("MSE")
    pl.plot(zoomXasis, zoomTest,label="Test MSE", color='red' )
    pl.plot(zoomXasis, zoomTrain, label="Training MSE",color='blue')
    pl.legend()
    #display.display(pl.gcf())

    pl.subplot(122 )
    #pl.cla()
    pl.title("All data")
    pl.xlabel("Epoc")
    pl.ylabel("MSE")
    pl.plot(mse_xaxis, mse_test_graph,  label="Test MSE", color='red' )
    pl.plot(mse_xaxis, mse_train_graph, label="Training MSE",color='blue')
    pl.legend()
    display.display(pl.gcf())
    
    for i in range (len(zoomTest)):
        print ("mse train: %s,  mse test: %s"  %(zoomTrain[i], zoomTest[i]))

def makePredictions(data_arr, label_arr, msg):
    mse, p = sess.run((cost,pred), feed_dict={x: data_arr,  y: label_arr, keep_prob: 1.0})
    pl.figure()
    pl.title(msg)
    pl.xlabel("Index")
    pl.ylabel("Prediction")
    pl.plot(p[0],label="Predicted Value", color='red' )
    pl.plot(label_arr[0], label="Actual Value",color='blue')
    pl.legend()
    display.display(pl.gcf())
    print "MSE "+str(mse)

for i in range(NUM_EPOCS):
    data_arr, label_arr, file_names = nextTrainingBatch(BATCH_SIZE)
    _, mse_train = sess.run((optimizer,cost), feed_dict={x: data_arr,  y: label_arr, keep_prob: dropout})
    if( i !=0 and i % TEST_AFTER == 0):
        testErrorAndReplot(i, mse_train)
    
       
for i in range(100):
    data_arr, label_arr, file_names = nextTestingBatch(1)
    makePredictions(data_arr, label_arr, str(i)+" Test Sample. File name: "+str(file_names[0]))
    
#for i in range(50):
#    data_arr, label_arr, file_names = nextTrainingBatch(1)
#    makePredictions(data_arr, label_arr, str(i)+" Training Sample. File name: "+str(file_names[0]))

        
print "Finished"




# In[ ]:




# In[ ]:



