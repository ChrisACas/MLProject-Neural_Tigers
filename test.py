import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets

from MNIST_Dataloader import MNIST_Dataloader
from sklearn.metrics import confusion_matrix
import pandas as pd
import math

lr = .0000000001 #learning rate
dbg = 0
pred_dbg = 0
class NeuralNetwork:
    def __init__(self, input_size=28*28, output_size=10, h_layers=1, h_neurons_per_layer=28):
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
        
        return np.array(desired_array)

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
    sigma = 1/(np.exp(np.multiply(-1., x)+1))
    return sigma*(1-sigma)

#Softmax
def softmax(x):
    exp_element = np.exp(x - np.max(x))
    return exp_element/np.sum(exp_element,axis=0)


def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))
    
def forward_backward_pass(x,y,nn,l_in,l_out):

    for i in range(0,len(x)):
        # transform image matrix flat array of values
        batch = 30
        sample=np.random.randint(0,x.shape[0],size=(batch))
        x_i=x[sample].reshape((-1,28*28))
        y_i=y[sample]

        # print(f'l_in: {l_in.shape}')
        # print(f'l_out: {l_out.shape}')
    
        # foward pass from input to hidden
        x_l_in = np.dot(x_i, l_in)
        x_in_sigmoid = sigmoid(x_l_in)
                
        # foward pass from hidden to output
        x_l_out = np.dot(x_in_sigmoid, l_out)
        out = softmax(x_l_out)

        # print(f'x_l_in: {x_l_in.shape}')
        # print(f'x_in_sigmoid: {x_in_sigmoid.shape}')
        # print(f'x_l_out: {x_l_out.shape}')
        # print(f'out: {out.shape}')
        # print(f'y_i: {y_i.shape}')

        #backpropogate output to hidden
        error = 2*(out-y_i) / out.shape[0] * d_softmax(x_l_out)
        delta_out = x_in_sigmoid.T @ error
        # print(f'error.shape: {error.shape}')
        # print(f'd_softmax(x_l_out) Shape: {d_softmax(x_l_out).shape}')
        # print(f'delta_out Shape: {delta_out.shape}')
        
        error = (l_out.dot(error.T).T * d_sigmoid(x_l_in))
        delta_in = x_i.T@error
        # print(f'error.shape: {error.shape}')
        # print(f'd_sigmoid(x_l_in) shape: {d_sigmoid(x_l_in).shape}')
        # print(f'delta_in.shape: {delta_in.shape}')

        l_out -= lr * ( delta_out )
        l_in -= lr * ( delta_in )
        # print(f'l_out Shape: {l_out.shape}')
        # print(f'l_in Shape: {l_in.shape}')
        # print(f'delta_out.shape: {delta_out.shape}')
        # print(f'delta_in.shape: {delta_in.shape}')


        # print(f'delta_in.shape: {delta_in}')
        # print(f'l_in.shape: {l_in.shape}')
        
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
    nn = NeuralNetwork()
    x_train, y_train = dataloader.get_train_data()

    # print(f'x_train.shape: {x_train.shape}')
    # print(f'y_train.shape: {y_train.shape}')
    # print(f'y_train.size: {y_train.shape}')

    y_train = np.array([nn.desired_array_out(y) for y in y_train])

    # print(f'y_train.shape: {y_train.shape}')

    for i in range(0,len(x_train)):
        for j in range(0,len(x_train[i])):
            x_train[i][j] = x_train[i][j]/255.
   
     # initialize input layer, hidden layer, output layer
    
    l_in = nn.layers[0]     # connection between input layer and hidden layer
    l_out = nn.layers[2]    # connection between hidden layer and output layer

    # print(f'input layer: {l_in.shape}')
    # print(f'output layer: {l_out.shape}')
    

    print('training started...')
    total_epochs = 10
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
