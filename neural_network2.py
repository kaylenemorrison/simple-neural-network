# -*- coding: utf-8 -*-


import numpy as np
import scipy.io as sc
import scipy.misc
import matplotlib.pyplot as plt
from numpy import random
#from sympy import init_printing
#from sympy import *

class Module():
    def __init__(self):
        self.prev = None # previous network (linked list of layers)
        self.next = None
        self.output = None # output of forward call for backprop.
        self.input = None

    #learning_rate = 1E-2 # class-level learning rate

    def link(self, input):
        self.prev = input
        self.prev.next = self

    def forward(self, input):
        raise NotImplementedError

    def backwards(self, input):
        raise NotImplementedError

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        

    def forward(self, input ):  ### input is a column vector
        # todo. compute sigmoid, update fields
        self.input = input
        exponential = np.exp(-1 * input) ### e^{-x} ## column vector
        ones = np.ones((input.shape[0], input.shape[1])) ## 1 , is a column vector
        denominator = ones + exponential ## 1+ e^{-x}
        z = np.divide(ones, denominator) ###1 /(1+e^{-x})
        self.output = z #### self.output is a column vector
    
    def backwards(self, gradient, learn_rate):
        sig_deriv = self.derivative()
        grad = np.multiply(gradient, sig_deriv) #### elementwise multiplication of gradient and derivative of sigmoid function
        return grad ## is a row vector
    
    def derivative(self):
        ones = np.ones((self.output.shape[0],self.output.shape[1])) ### column vector like self.output
        sub = np.subtract(ones, self.output) ## 1- sig(x). ## column vector
        deriv = np.multiply(self.output,sub) ### sig(x)(1-sig(x)) ### column vector
        deriv = deriv.T ### deriv is now a row vector
        return deriv ### row vector
    
    def sanity_check(self):
        print("Adding a sigmoid layer with dimension: " + str(self.prev.outputdim))

class Linear(Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        # todo. initialize weights and biases. 
        self.inputdim = input_size
        self.outputdim = output_size
        self.weights = np.random.rand(output_size,input_size)
        self.biases = np.zeros((output_size,1))
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, input):  
        self.input = input
        self.output = np.add(np.matmul(self.weights,input), self.biases)

    def updateVars(self,learn_rate):
      
        self.weights = np.subtract(self.weights, learn_rate * self.weights_grad )
        self.biases = np.subtract(self.biases,learn_rate * self.biases_grad) ## self.biases_grad.T is a column vector

    def backwards(self, gradient, learn_rate):  ## gradient is a row vector
       
       
        self.weights_grad = (np.matmul(self.input, gradient)).T ## changed to self.input because self.prev of first layer is NoneType
        self.biases_grad = gradient.T  ## change self.biases_grad to a column vector
        grad = np.matmul(gradient, self.weights) ### gradient is (1 by output dimension) wieghts is (output by input)
        self.updateVars(learn_rate)
       
        return grad ### 1 by self.prev.output in dimension
        
   
    def sanity_check(self):
        print("Adding a linear layer with input dimension: " + str(self.inputdim) + " and output dimension: " + str(self.outputdim))

## overall neural network class
class Network(Module):
    def __init__(self, layers_array):
        super(Network, self).__init__()
        # todo initializes layers, i.e. sigmoid, linear
        self.layers_array = layers_array
        self.first_layer = None
        self.last_layer = None
        
        for i in range(len(self.layers_array)-1):
          if i == 0: ### if this is the first layer, no previous layer to link to
            linear = Linear(layers_array[i],layers_array[i+1])
            self.first_layer = linear
            sigmoid = Sigmoid()
            sigmoid.link(linear)
          elif i == (len(layers_array)-2):  #### make the last layer a linear layer
            linear = Linear(layers_array[i],layers_array[i+1])
            linear.link(sigmoid)
            self.last_layer = linear
          else:  ## initializing all of the layers in between
            linear = Linear(layers_array[i],layers_array[i+1])
            linear.link(sigmoid)
            sigmoid = Sigmoid()
            sigmoid.link(linear)


    def forwardprop(self, input):
        ## iterate through layers, do forward propagation
        layer = self.first_layer
        while(layer != None):
          layer.forward(input)
          input = layer.output
          layer = layer.next
        return self.last_layer.output
        

    def backprop(self, grad, learn_rate):   ### grad parameter passed in from mse backward
        layer = self.last_layer
        while(layer != None):
          #print(grad)
          grad = layer.backwards(grad,learn_rate)
          layer = layer.prev


    def predict(self, data):  ### for computing a prediction on a single training example
        # todo compute forward pass and output predictions
        prediction = self.forwardprop(data)
        return prediction

    def mse_backward(self, test_labels):  ### test labels are a column vector
        grad = 2* (np.subtract(self.last_layer.output,test_labels)) ## error has dimensions of nodes in final linear layer X 1
        grad.reshape((1,grad.shape[0])) ## reshape to a row vector
        return grad ## this is the derivative of the mse (it is the gradient)

    def mse_forward(self,input,test_labels):
        subtract = np.subtract(input, test_labels) ### column vector
        squared_error = subtract @ subtract.T ## find the squared error.  (dot product (1 by something) (something by 1))
        mse = squared_error/(len(subtract)) ## find the mean squared error
        return mse ### scalar value, used for keeping track of accuracy
      

    #def accuracy(self, test_data, test_labels):
        # todo evaluate accuracy of model on a test dataset
        #pass

    #def loss(input, test_data):
        #pass

#### STOCHASTIC GRADIENT DESCENT

# function for training the network for a given number of iterations

def train(model, data, labels, epochs,  learning_rate):
    # todo repeatedly do forward and backwards calls, update weights, do 
    # stochastic gradient descent on mini-batches.

    for epoch in range(epochs): ### number of passes through training data
      mse_arr = []
      for i in range(len(data)):  ### iterate through training data and test labels, doing stochastic gradient descent
        prediction = model.forwardprop(data[i])
        mse = model.mse_forward(prediction,labels[i])
        mse_arr.append(mse)
        grad_mse_backward = model.mse_backward(labels[i])
        model.backprop(grad_mse_backward, learning_rate)

        if ((epoch % 10 == 0) and (i == (len(data) -1)) ):  ## every 10 epochs, at the end of a full pass through the training data
          mse_mean = np.mean(mse_arr)
          print("total error: " + str(mse_mean))

##### predicting output 

def predict_output(model,data):  ### for computing a prediction for all training data
    output = []
    for i in range(len(data)):
      prediction = model.predict(data[i])
      #print(prediction)
      output.append(prediction)

    return(output)


def main():
    #### create the network for xor 

  network = Network([2,3,1]) ## neural network with one input node, a hidden layer with 3 nodes, one output nodes
  layer = network.first_layer
  while(layer != None):
    layer.sanity_check()
    layer = layer.next

  #### train on xor
  xor1 = [0,0]
  xor2 = [1,0]
  xor3 = [0,1]
  xor4 = [1,1]

  data = [np.array(xor1).reshape((2,1)), np.array(xor2).reshape((2,1)),np.array(xor3).reshape((2,1)), np.array(xor4).reshape((2,1))]
  #print(data[0].shape)
  labels = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]
  #print(labels[0].shape)

  train(network, data, labels, 500, 0.23)
#### prediction on xor

  output = predict_output(network, data )
  #print(type(data))
  print("\n")
  print("Output \n")
  print(data[0][0][0], "XOR",data[0][1][0], "=",output[0][0][0], "\n")
  print(data[1][0][0], "XOR",data[1][1][0], "=",output[1][0][0], "\n")
  print(data[2][0][0], "XOR",data[2][1][0], "=",output[2][0][0], "\n")
  print(data[3][0][0], "XOR",data[3][1][0], "=",output[3][0][0], "\n")

 
  #print(type(output))
 # print(output)
  #print(data[1])



if __name__ == "__main__":
    main()




