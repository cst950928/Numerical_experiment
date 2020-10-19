# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:14:43 2020

@author: DHU
"""

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle
from  sklearn.svm  import LinearSVC
from sklearn import neighbors 
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score  
import pandas as pd
from openpyxl import load_workbook


#mnist_F-cluSL
def read_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot = False)
    
    # The number of training and testing samples
    train_nums = mnist.train.num_examples
    #validation_nums = mnist.validation.num_examples
    test_nums = mnist.test.num_examples
    
    
    #The value of training and testing samples
    train_data = mnist.train.images   #traning samples (55000, 784)
    #val_data = mnist.validation.images  #(5000,784)
    test_data = mnist.test.images       #testing samples (10000,784)
    
    
    #The ys of training and testing samples
    train_labels = mnist.train.labels     #(55000,10)
    train_labels = train_labels + 1
    #val_labels = mnist.validation.labels  #(5000,10)
    test_labels = mnist.test.labels       #(10000,10)
    test_labels = test_labels + 1
    
    
    return train_nums, test_nums, np.shape(train_data)[1], train_data, train_labels, max(train_labels), test_data, test_labels


#mnist_CNN
def read_mnist_onehot():
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
    
    # The number of training and testing samples
    train_nums = mnist.train.num_examples
    #validation_nums = mnist.validation.num_examples
    test_nums = mnist.test.num_examples
    
    
    #The value of training and testing samples
    train_data = mnist.train.images   #traning samples (55000, 784)
    #val_data = mnist.validation.images  #(5000,784)
    test_data = mnist.test.images       #testing samples (10000,784)
    
    
    #The ys of training and testing samples
    train_labels = mnist.train.labels     #(55000,10)
    #val_labels = mnist.validation.labels  #(5000,10)
    test_labels = mnist.test.labels       #(10000,10)

    
    return train_nums, test_nums, np.shape(train_data)[1], train_data, train_labels, np.shape(train_labels)[1], test_data, test_labels



#cifar-10
def unpickle(file):
    with open (file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

def read_cifar_10(file_test, file_train1, file_train2, file_train3, file_train4, file_train5):
    
    test_data = unpickle(file_test)
    vay = np.array(test_data[b'labels'])
    temp_vax = np.array(test_data[b'data'])
    
    train_data = unpickle(file_train1)
    y = np.array(train_data[b'labels'])
    temp_x = np.array(train_data[b'data'])
    
    train_data = unpickle(file_train2)
    y = np.r_[y, np.array(train_data[b'labels'])]
    temp_x = np.r_[temp_x, np.array(train_data[b'data'])]
    
    train_data = unpickle(file_train3)
    y = np.r_[y, np.array(train_data[b'labels'])]
    temp_x = np.r_[temp_x, np.array(train_data[b'data'])]
    
    train_data = unpickle(file_train4)
    y = np.r_[y, np.array(train_data[b'labels'])]
    temp_x = np.r_[temp_x, np.array(train_data[b'data'])]
    
    train_data = unpickle(file_train5)
    y = np.r_[y, np.array(train_data[b'labels'])]
    temp_x = np.r_[temp_x, np.array(train_data[b'data'])]
    
    
    y = y + 1
    vay = vay + 1
    
    x_Red = 0.299 * temp_x[:, range(1024)]
    x_Green = 0.587 * temp_x[:, range(1024, 2048)]
    x_Blue = 0.114 * temp_x[:, range(2048, 3072)]
    
    vax_Red = 0.299 * temp_vax[:, range(1024)]
    vax_Green = 0.587 * temp_vax[:, range(1024, 2048)]
    vax_Blue = 0.114 * temp_vax[:, range(2048, 3072)]
    
    x = x_Red + x_Green + x_Blue
    vax = vax_Red + vax_Green + vax_Blue

    return np.shape(x)[0], np.shape(x)[1], max(y), np.shape(vax)[0], x, y, vax, vay

    

import os

class BatchRename():

    def __init__(self):
        self.path = '.../data/Magnetic-tile-defect-datasets.-master/MT_Break/black_2500'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i) + '.png')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))
    
        
    def resize(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            img = Image.open(self.path + '/' + str(i) + '.png')
            img = img.resize((50, 50), Image.ANTIALIAS)
            
            savename = os.path.join(self.path, str(i) + '.png')
            img.save(savename)
            i = i + 1
            
        
    def imagetoarray(self):
        filelist = os.listdir(self.path)
        i = 0
        filename = '.../MT.txt'

        total_data = np.zeros((103, 10000))
        for item in filelist:
            img = Image.open(self.path + '/' + str(i) + '.png')
            img = np.array(img)
            new_im = Image.fromarray(img)
#            new_im.show()
            plt.imshow(new_im)
            plt.show()
            img = img.reshape(1, -1)
#            img = img/255
            total_data[i, :] = img

            i = i + 1
            
        np.savetxt(filename, total_data, fmt = '%d')
        
    def read_excel_data(self):
        filename = '.../MT.xlsx'
        df = pd.read_excel(filename)
        data = df.iloc[0:2,:]
        
        print(np.shape(data))
        
        print(data)
        
        
        
        
if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
    demo.resize()
#    demo.imagetoarray()
#    demo.read_excel_data()



def imagetoarray(filename, N, training_num):
    
    total_data = np.zeros((N, 2500))
    
    for i in range(N):
        img = Image.open(filename + '/' + str(i) + '.png')
        img = np.array(img)
        img = img.reshape(1, -1)
        total_data[i, :] = img
    
    total_x = total_data[range(training_num), :]
    total_vax = total_data[range(training_num, N), :]
      
    return total_x, total_vax




