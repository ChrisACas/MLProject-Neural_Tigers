import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets

from MNIST_Dataloader import MNIST_Dataloader
from sklearn.metrics import confusion_matrix
import pandas as pd

lr = .001 #learning rate
dbg = 0
pred_dbg = 0
class NeuralNetwork:
    def __init__(self, input_size=28*28, output_size=10, h_layers=1, h_neurons_per_layer=128):
        self.input_size = input_size
        self.output_size = output_size
        self.h_layers = h_layers
        self.h_neurons_per_layer = h_neurons_per_layer
        self.layers = self.init_layers(input_size, h_neurons_per_layer, h_layers, output_size)

    # TODO: implement a programmable amount of hidden layer initialization
    def init_layers(self, input_size, h_neurons_per_layer, h_layers, output_size):
        '''
        Get layer size info and develop weight array 
        initialize random weights for each connection to next layer
            weight array of output size, in array for every input node 
        return these weight arrays for each node as layer
        '''
        '''layer1 = np.random.uniform(-1.,1.,size=(input_size, h_neurons_per_layer))\
            /np.sqrt(input_size * h_neurons_per_layer)
        
        layer2 = np.random.uniform(-1.,1.,size=(h_neurons_per_layer, output_size))\
            /np.sqrt(h_neurons_per_layer * output_size)'''

        layer_in = np.random.randn(input_size, h_neurons_per_layer)

        #hlayers = []
        #for i in range(0,h_layers):
        #    hlayers[i] = np.random.randn(h_neurons_per_layer, h_neurons_per_layer)

        layer_out = np.random.randn(h_neurons_per_layer, output_size)
        
        return [layer_in,  layer_out]
    
    def desired_array_out(self, label):
        '''Turn label into desired output array 
        input label         5
        return desire array [0 0 0 0 0 1 0 0 0 0]
        '''
        desired_array = np.zeros(self.output_size, np.float32)
        desired_array[label] = 1
        
        return desired_array

#Sigmoid funstion
# def sigmoid(x):
#     x_modified = [-5. if ele < -100.0 else ele for ele in x]
#     x_modified = [5. if ele > 100.0 else ele for ele in x_modified]
#     # print(x_modified)
#     x_modified = np.array(x_modified)
#     sigma = 1 / (np.exp(np.multiply(-1., np.asarray(x_modified)) + 1))
#     # return sigma
#     return 1/(np.exp(-x)+1)

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

#derivative of sigmoid
def d_sigmoid(x):
    #modified_x = []
    #print(x)
    x_modified = [-5. if ele < -100.0 else ele for ele in x]
    x_modified = [5. if ele > 100.0 else ele for ele in x_modified]
    x_modified = np.array(x_modified)
    if dbg:
        print(x_modified)
    sigma = 1/(np.exp(np.multiply(-1.,np.asarray(x_modified))+1))
    # if max(sigma) < 1.0e-6:
    #     return np.zeros(len(x_modified), np.float32)
    # else:
    #     return sigma*(1-sigma)
    return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax
def softmax(x):
    x = [-5. if ele < -100.0 else ele for ele in x]
    x = [5. if ele > 100.0 else ele for ele in x]
    x = np.array(x)
    exp_element = np.exp(x -x.max())
    #exp_element=np.exp(x)#-x.max())
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):
    x = [-5. if ele < -100.0 else ele for ele in x]
    x = [5. if ele > 100.0 else ele for ele in x]
    x = np.array(x)
    exp_element = np.exp(x -x.max())
    #exp_element=np.exp(x)#-x.max())
    sigma = exp_element/np.sum(exp_element,axis=0)
    return (sigma)*(1-sigma)


    #forward and backward pass
def forward_backward_pass(x,y,nn,l1,l2):
 
    for i in range(0,len(x)):
        x_i=x[i]
        y_i=nn.desired_array_out(y[i])
        

        #forward pass

        # check input image matrices and output arrays sahpe
        if dbg: 
            print(f'x_i len: {len(x_i[0])}')
            print(f'y_i len: {len(y_i)}')

        # check shape of layers
        if dbg:
            print('l1 shape = {}'.format(l1.shape))
            print('l2 shape = {}'.format(l2.shape))
            
        x_l1 = np.dot(l1.T,np.array(x_i).flatten())
        if dbg:
            print('x_l1 shape={}'.format(x_l1.shape))
        x_sigmoid = sigmoid(x_l1)
        if dbg:
            print('x_sigmoid shape={}'.format(x_sigmoid.shape))

        x_l2 = np.dot(x_sigmoid.T, l2)
        if dbg:
            print('x_l2 shape={}'.format(x_l2.shape))
        out = softmax(x_l2)
        if dbg:
            print('out shape={}'.format(out.shape))
            print('out={}'.format(out))

        # backward pass
        error=y_i-out
    
        delta2 = lr * error * d_softmax(x_l2)
        if dbg:
            print('delta2={}'.format(delta2))
            print('before updation l2 sum={}'.format(np.sum(l2)))
            #print('x_sigmoid shape={}'.format(x_sigmoid.shape))
            print('delta2.T shape={}'.format(delta2.T.shape))
        
        l2 = np.add(l2, np.outer(x_sigmoid,delta2))#delta2.T * x_sigmoid)
        if dbg:
            print('after updation l2 sum={}'.format(np.sum(l2)))

            #delta1 = ((l2[:, neuron_i]).dot(error.T)).T * d_sigmoid(x_l1)
            print('d_sigmoid(x_l1) shape={}'.format(d_sigmoid(x_l1).shape))
        
        delta1 = ((l2).dot(error.T)).T * d_sigmoid(x_l1)
        if dbg:
            print('delta1 shape={}'.format(delta1.shape))
            print('x_i.T shape={}'.format((np.array(x_i).flatten()).T.shape))
        
        l1 = np.add(l1, np.outer(np.array(x_i).flatten(),delta1))
        
    return out,l1,l2

def predict(x,y,nn,l1,l2,y_test,y_pred):
    targets = np.zeros((len(y),10), np.float32)
    targets[range(targets.shape[0]),y] = 1


    x_sigmoid = np.zeros(l2.shape[0], np.float32)

    # forward pass
    for i in range(0,len(x)):
        #if i%50000 == 0:
        #    print('i = {}'.format(i))
        x_i=x[i]
        y_test.append(y[i])
        y_i=nn.desired_array_out(y[i])

        x_l1 = np.dot(l1.T,np.array(x_i).flatten())
        if dbg:
            print('x_l1 shape={}'.format(x_l1.shape))
        x_sigmoid = sigmoid(x_l1)
        if dbg:
            print('x_sigmoid shape={}'.format(x_sigmoid.shape))

        x_l2 = np.dot(x_sigmoid.T, l2)
        if dbg:
            print('x_l2 shape={}'.format(x_l2.shape))
        out = softmax(x_l2)
        predicted_num = np.argmax(out)
        if pred_dbg:
            #print('out shape={}'.format(out.shape))
            print('out={}'.format(out))
            print('predicted_num={} actual_num={}'.format(predicted_num,y[i]))
            #print('y_i={}'.format(y_i))
        y_pred.append(predicted_num)

def get_input(x_train, y_train, idx):
    return x_train[idx], y_train[idx]

def main():

    # load data
    dataloader = MNIST_Dataloader()
    x_train, y_train = dataloader.get_train_data()
  
    # make layers
    nn = NeuralNetwork()
    l1 = nn.layers[0]
    l2 = nn.layers[1]

    print('training started...')
    total_epochs = 1
    for epoch in range(0,total_epochs):
        print(f'epoch={epoch}')
        out,l1,l2 = forward_backward_pass(x_train,y_train,nn,l1,l2)

    y_test = []
    y_pred = []


    print('validation started...')
    predict(x_train,y_train,nn,l1,l2,y_test,y_pred)

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
    #sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    #plt.show()


  
if __name__=="__main__":
    main()
