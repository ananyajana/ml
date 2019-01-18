#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:22:06 2018

@author: aj611
"""

import torch
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time

#torch.set_printoptions(precision=10)

epochs = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logfile_name = 'log_file_gpu1_ep300_different_conv_params_200_neurons.txt'
log_statement = "This is CNN test on gpu CIFAR10 database for a different set of parameters.\n"
f = open(logfile_name, 'w+')
f.write(log_statement)
f.write("number of epochs: {}".format(epochs))
f.write("machine: {}".format(device))


f.write("############ Convolutional layer1 parameters:##############\n")
conv1_num_input_channels = 3
#conv1_num_output_channels = 6
conv1_num_output_channels = 50
conv1_kernel_size = 5

f.write("number of input channels: {}\n".format(conv1_num_input_channels))
f.write("number of output channels: {}\n".format(conv1_num_output_channels))
f.write("kernel size: {}\n".format(conv1_kernel_size))

f.write("############ Convolutional layer2 parameters:##############\n")
#conv2_num_input_channels = 6
conv2_num_input_channels = 50
#conv2_num_output_channels = 16
conv2_num_output_channels = 150
conv2_kernel_size = 5

f.write("number of input channels: {}\n".format(conv2_num_input_channels))
f.write("number of output channels: {}\n".format(conv2_num_output_channels))
f.write("kernel size: {}\n".format(conv2_kernel_size))

f.write("############ Linear layer1 parameters:##############\n")
#lin1_num_channels = 16
lin1_num_channels = 150
lin1_height = 5
lin1_width = 5
lin1_matrix_rows = lin1_num_channels * lin1_height* lin1_width
lin1_matrix_cols =  120

f.write("lin1_matrix_rows: {}\n".format(lin1_matrix_rows))
f.write("lin1_matrix_cols: {}\n".format(lin1_matrix_cols))


f.write("############ Linear layer2 parameters:##############\n")
lin2_matrix_rows =  120
lin2_matrix_cols =  84

f.write("lin2_matrix_rows: {}\n".format(lin2_matrix_rows))
f.write("lin2_matrix_cols: {}\n".format(lin2_matrix_cols))


f.write("############ Linear layer3 parameters:##############\n")
lin3_matrix_rows =  84
lin3_matrix_cols =  10

f.write("lin3_matrix_rows: {}\n".format(lin3_matrix_rows))
f.write("lin3_matrix_cols: {}\n".format(lin3_matrix_cols))

f.write("############ Maxpooling layer1 parameters:##############\n")
maxpool1_kernel_rows =  2
maxpool1_kernel_cols =  2

f.write("############ Maxpooling layer2 parameters:##############\n")
maxpool2_kernel_rows =  maxpool2_kernel_cols =  2

f.write("############ Net Architecture:##############\n")
f.write("conv1---->relu--->maxpool---->conv2---->relu---->maxpool----?Linear1---->relu---->Linear2---->relu---->Linear3\n")

f.write("############ Learning parameters:##############\n")
learning_rate = 0.001
momentum = 0.2
f.write("learning_rate: {}\n".format(learning_rate))
f.write("momentum: {}\n".format(momentum))

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel,6 output channels, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #f.write("The parameters of the first convolutional neural network is:\n")
        
        self.conv1 = nn.Conv2d(conv1_num_input_channels, conv1_num_output_channels, conv1_kernel_size)
        
        # 6 input channels, 16 output channels, 5x5 convolution square kernel
        self.conv2 = nn.Conv2d(conv2_num_input_channels, conv2_num_output_channels, conv2_kernel_size)
        
        #an affine operation: y = Wx +  b
        #linear transformation of a 1x(16 * 5* 5) matrix into a 1x120 matrix
        self.fc1 = nn.Linear(lin1_matrix_rows, lin1_matrix_cols)
        
        #random biases??
        #print("fc1 bias")
        #print(self.fc1.bias.size())
        #print(self.fc1.bias)
        
        #linear transformation of a 1x120 matrix into a 120x84 matrix
        self.fc2 =nn.Linear(lin2_matrix_rows, lin2_matrix_cols)
        #linear transformation of a 120x84 matrix into a 84x10 matrix
        self.fc3 = nn.Linear(lin3_matrix_rows, lin3_matrix_cols)
        
    def forward(self, x):
        # Max pooling over a 2x2 window
        #f.write("First convolutional and maxpool layer 1:\n")
        x = F.max_pool2d(F.relu(self.conv1(x)), (maxpool1_kernel_rows, maxpool1_kernel_cols))
        
        #if the size is a square you can only specify a single number
        #f.write("Second convolutional and maxpool layer 2:\n")
        x = F.max_pool2d(F.relu(self.conv2(x)), maxpool2_kernel_rows)
        #f.write("Flattening the array\n")
        x = x.view(-1, self.num_flat_features(x))
        #f.write("Applying relu on first linear layer\n")
        x = F.relu(self.fc1(x))
        #f.write("Applying relu on second linear layer\n")
        x = F.relu(self.fc2(x))
        #f.write("Applying the third linear layer1\n")
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        #print(x.size())
        #print("hello")
        #print(size)
        
        num_features = 1
        for s in size:
            num_features *= s
        #print(num_features)
        #f.write("flattening the array\n")
        return num_features
    
net = Net()
print(net)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net.to(device)
f.write("setting the device to gpu/cpu \n")



params = list(net.parameters())
print(len(params))
print(params[0].size())


input = torch.randn(1, 1, 32, 32)
#out = net(input)
#print(out)

#net.zero_grad()
#out.backward(out)

f.write("Normalize the images")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


f.write("Load training data\n")
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)

f.write("Load test data:\n")
#f.write("Testloader params: train = {}, download = {T\n")
testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'ship-truck')


def imshow(img):
    # need to rever the normalization process to get the original image
    img = img / 2 + 0.5
    #f.write("sending the training images, labels to gpu")
    #images, labels = images.to(device),labels.to(device)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
######## commenting out this part while running in CLI################3

f.write("tic starts\n")    
tic = time.time()
dataiter = iter(trainloader)
images, labels= dataiter.next()
print(images)
print(labels)
#f.write("sending the training images, labels to gpu")
#images, labels = images.to(device),labels.to(device)


imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)

tic = time.time()
#training
f.write("training starts`")
for epoch in range(epochs):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        #f.write("sending the training images, labels of epoch to gpu")
        inputs, labels = inputs.to(device),labels.to(device)

        optimizer.zero_grad()
        
        output = net(inputs)
        loss = criterion(output, labels)
        #loss = criterion(output, labels).double()
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #running_loss += loss.item().double()
        
        if i % 2000 == 1999:
            buf = '[%d %5d] loss %.10f' % (epoch + 1, i + 1, running_loss / 2000)
            print(buf)
            #print(buf)
            f.write(buf)
            f.write("\n")
            running_loss = 0.0
print("Finished training\n")
f.write("Finished training\n") 
toc = time.time()
print('Time for training: ' , toc - tic)
f.write('Time for training: {0}'.format(toc - tic))
tic = time.time()

#class_correct = list(0. for i in range(10))
#class_total = list(0. for i in range(10))

#creating the confusion matrix i.e. image Vs which class it is mapped to 
f.write("Creating the confusion matrix. \n")
confusion_matrix = np.zeros((len(classes), len(classes)))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        #f.write("sending the test images, labels to gpu")
        images, labels = images.to(device),labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        #c = (predicted == labels).squeeze()
        for i in range(4):
            confusion_matrix[labels[i], predicted[i]] += 1 


f.write("############# Confusion matrix #############\n      ")
for c in classes:
    f.write("{0}  ".format(c))
f.write('\n')

for indx, row in enumerate(confusion_matrix):
    f.write(classes[indx] +  "T ")
    for val in row:
        f.write("{0} ".format(int(val)))
    f.write('\n')
f.write("############# ------------------ #############\n   ")    
        
toc = time.time()
print('Time for Testing: ' , toc - tic)
f.write('Time for Testing: {0}'.format(toc - tic))

f.close()
