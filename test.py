import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets

from MNIST_Dataloader import MNIST_Dataloader
from sklearn.metrics import confusion_matrix
import pandas as pd
import math

lr = .00000000000001 #learning rate
dbg = 0
pred_dbg = 0
class NeuralNetwork:
    def __init__(self, input_size=28*28, output_size=10, h_layers=1, h_neurons_per_layer=16):
        self.input_size = input_size
        self.output_size = output_size
        self.h_layers = h_layers
        self.h_neurons_per_layer = h_neurons_per_layer
        self.layers = self.init_layers(input_size, h_neurons_per_layer, h_layers, output_size)

    # TODO: implement a programmable amount of hidden layer initialization
    def init_layers(self, input_size, h_neurons_per_layer, h_layers, output_size):
        layer_in = np.random.uniform(-1.,1.,size=(input_size, h_neurons_per_layer))
        
        hl = []
        for _ in range(0, h_layers):
            hl.append(np.random.uniform(-1.,1.,size=(h_neurons_per_layer, h_neurons_per_layer)))

        #layer_out = np.zeros((h_neurons_per_layer, output_size),np.float32)
        layer_out = np.random.uniform(-1.,1.,size=(h_neurons_per_layer, output_size))
        return [layer_in, hl, layer_out]
    
    def desired_array_out(self, label):
        '''Turn label into desired output array 
        input label         5
        return desire array [0 0 0 0 0 1 0 0 0 0]
        '''
        desired_array = np.zeros(self.output_size, np.float32)
        desired_array[label] = 1
        
        return desired_array

#Sigmoid funstion
def sigmoid(x):
    try:
        res = 1 / (1 + np.exp(-x))
    except OverflowError:
        res = 0.0
    return res
    #return 1/(np.exp(-x)+1)

#derivative of sigmoid
def d_sigmoid(x):
    x[x<-100.] = -5.
    x[x > 100.] = 5.
  
    if dbg:
        print(x)
    sigma = 1/(np.exp(np.multiply(-1., x)+1))
    return sigma*(1-sigma)
    #return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax
def softmax(x):
    x_modified = [-5. if ele < -100.0 else ele for ele in x]
    x_modified = [5. if ele > 100.0 else ele for ele in x_modified]
    x_modified = np.array(x_modified)
    exp_element = np.exp(x_modified - np.max(x_modified))
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):

    # print(f'x:{x}')
    # SM = x.reshape((-1, 1))
    # jac = np.diagflat(x) - np.dot(SM, SM.T)
    # print(f'jac:{jac}')
    # return jac
    x_modified = np.array([-5. if ele < -100.0 else ele for ele in x])
    x_modified = np.array([5. if ele > 100.0 else ele for ele in x_modified])
    return x_modified * np.identity(x_modified.size) - x_modified.transpose() @ x_modified


    
def forward_backward_pass(x,y,nn,l_in,l_out):
    # x_sigmoid = np.zeros(l_out.shape[0], np.float32)
    # x_sigmoid_hl = []
    # for i in range(0,nn.h_layers):
    #     x_sigmoid_hl.append(np.zeros(l_in.shape[1], np.float32))
   
    # forward pass
    for i in range(0,len(x)):
        x_i=x[i]
        y_i=nn.desired_array_out(y[i])


        # foward pass from input to hidden
        x_l_in = np.dot(l_in.T, np.array(x_i).flatten())
        x_in_sigmoid = sigmoid(x_l_in)
        
        # print(f'x_in_sigmoid: {x_in_sigmoid}')

        
        # foward pass from hidden to output
        x_l_out = np.dot(x_in_sigmoid.T, l_out)
        out = softmax(x_l_out)

        # print(f'x_l_out: {x_l_out}')
        # print(f'out: {out}')

        #backpropogate output to hidden
        error = np.power(y_i-out,2).mean()
        delta_out = 2 * error * d_softmax(x_l_out)
        l_out -= lr * ( x_l_out.T @ delta_out )


        # print(f'error: {error}')
        # print(f'delta_out.shape: {delta_out}')
        # print(f'l_out.shape: {l_out.shape}')

        #backpropogate hidden to input
        delta_in = 2 * error * d_sigmoid(x_in_sigmoid)
        l_in -= lr * ( x_l_in.T @ delta_in )

        # print(f'delta_in.shape: {delta_in}')
        # print(f'l_in.shape: {l_in.shape}')

        '''
        # for hl_index in reversed(range(0,nn.h_layers)):
        #     if hl_index == nn.h_layers-1:
        #         delta_hl = ((l_out).dot(delta_out.T)).T * d_sigmoid(x_sigmoid_hl[hl_index])
        #         hl[hl_index] = np.add(hl[hl_index],-1.*lr*delta_hl)
        #     else:
        #         delta_hl = ((hl[hl_index+1]).dot(delta_hl.T)).T * d_sigmoid(x_sigmoid_hl[hl_index])
        #         hl[hl_index] = np.add(hl[hl_index], -1.*lr * delta_hl)

        #delta_in = ((hl[0]).dot(delta_hl.T)).T * d_sigmoid(x_l_in)
    
        #l_in = np.add(l_in, np.outer(np.array(x_i).flatten(),-1.*lr*delta_in))
        '''
    print()
    print(f'l_out: {l_out}')
    return l_in,l_out

def predict(x,y,nn,l_in,l_out,y_test, y_pred):
    # forward pass
    for i in range(0,len(x)):
        x_i=x[i]
        y_i=nn.desired_array_out(y[i])


        # foward pass from input to hidden
        x_l_in = np.dot(l_in.T, np.array(x_i).flatten())
        x_in_sigmoid = sigmoid(x_l_in)
        
        # print(f'x_in_sigmoid: {x_in_sigmoid}')

        
        # foward pass from hidden to output
        x_l_out = np.dot(x_in_sigmoid.T, l_out)
        out = softmax(x_l_out)

        # print(f'x_l_out: {x_l_out}')
        # print(f'out: {out}')

        predicted_num = np.argmax(out)
        y_pred.append(predicted_num)
        y_test.append(y[i])

def get_input(x_train, y_train, idx):
    return x_train[idx], y_train[idx]

def main():

    # Get training data and normalize values in matrix
    dataloader = MNIST_Dataloader()
    x_train, y_train = dataloader.get_train_data()
    for i in range(0,len(x_train)):
        for j in range(0,len(x_train[i])):
            x_train[i][j] = x_train[i][j]/255.
   

     # initialize input layer, hidden layer, output layer
    nn = NeuralNetwork()
    l_in = nn.layers[0]     # connection between input layer and hidden layer
    l_out = nn.layers[2]    # connection between hidden layer and output layer

    print(f'input layer: {l_in.shape}')
    print(f'output layer: {l_out.shape}')
    

    print('training started...')
    total_epochs = 5
    for epoch in range(0,total_epochs):
        print('epoch={}'.format(epoch))
        l_in,l_out = forward_backward_pass(x_train,y_train,nn,l_in,l_out)

    y_test = []
    y_pred = []


    print('validation started...')
    predict(x_train,y_train,nn,l_in,l_out,y_test,y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                         columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    tp = 0
    tot_cnt = len(y_test)
    for j in range(0,len(y_test)):
        if y_test[j] == y_pred[j]:
            tp += 1
    print('Precision = {0:.3f}'.format(tp/(1.*tot_cnt)))

    # Plotting the confusion matrix
    plt.figure(figsize=(5, 4))
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    


  
if __name__=="__main__":
    main()
