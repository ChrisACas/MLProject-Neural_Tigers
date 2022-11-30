import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets

from MNIST_Dataloader import MNIST_Dataloader
from sklearn.metrics import confusion_matrix
import pandas as pd
import math

lr = 0.05 #learning rate
dbg = 0
pred_dbg = 0
class NeuralNetwork:
    def __init__(self, input_size=28*28, output_size=10, h_layers=1, h_neurons_per_layer=3):
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

        hl = []

        '''layer_in = np.random.rand(input_size, h_neurons_per_layer)

        for i in range(0, h_layers):
            hl.append(np.random.rand(h_neurons_per_layer, h_neurons_per_layer))

        layer_out = np.random.rand(h_neurons_per_layer, output_size)'''

        layer_in = np.random.uniform(-1.,1.,size=(input_size, h_neurons_per_layer))

        for i in range(0, h_layers):
            hl.append(np.random.uniform(-1.,1.,size=(h_neurons_per_layer, h_neurons_per_layer)))

        #layer_out = np.random.uniform(-1.,1.,size=(h_neurons_per_layer, output_size))
        layer_out = np.zeros((h_neurons_per_layer, output_size),np.float32)
        
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
    '''x_modified = [-5. if ele < -100.0 else ele for ele in x]
    x_modified = [5. if ele > 100.0 else ele for ele in x_modified]
    # print(x_modified)
    x_modified = np.array(x_modified)
    sigma = 1 / (np.exp(np.multiply(-1., np.asarray(x_modified)) + 1))
    return sigma'''
    try:
        res = 1 / (1 + np.exp(-x))
    except OverflowError:
        res = 0.0
    return res
    #return 1/(np.exp(-x)+1)

#derivative of sigmoid
def d_sigmoid(x):
    #modified_x = []
    #print(x)
    x[x<-100.] = -5.
    x[x > 100.] = 5.
    #x_modified = [-5. if ele < -100.0 else ele for ele in x]
    #x_modified = [5. if ele > 100.0 else ele for ele in x_modified]
    #x_modified = np.array(x_modified)
    if dbg:
        print(x)
    sigma = 1/(np.exp(np.multiply(-1., x)+1))
    '''if max(sigma) < 1.0e-6:
        return np.zeros(len(x), np.float32)
    else:'''
    return sigma*(1-sigma)
    #return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax
def softmax(x):
    x_modified = [-5. if ele < -100.0 else ele for ele in x]
    x_modified = [5. if ele > 100.0 else ele for ele in x_modified]
    x_modified = np.array(x_modified)
    exp_element = np.exp(x_modified - np.max(x_modified))
    #exp_element = np.exp(x_modified)
    #exp_element=np.exp(x)#-x.max())
    #expA = np.exp(A)
    #return exp_element / exp_element.sum(axis=1, keepdims=True)
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):
    '''x_modified = [-5. if ele < -100.0 else ele for ele in x]
    x_modified = [5. if ele > 100.0 else ele for ele in x_modified]
    x_modified = np.array(x_modified)
    #exp_element = np.exp(x_modified -x_modified.max())
    exp_element=np.exp(x_modified)#-x.max())
    sigma = exp_element/np.sum(exp_element,axis=0)
    return (sigma)*(1-sigma)'''
    SM = x.reshape((-1, 1))
    jac = np.diagflat(x) - np.dot(SM, SM.T)
    return jac


    #forward and backward pass
def forward_backward_pass(x,y,nn,l_in,hl,l_out):
    targets = np.zeros((len(y),10), np.float32)
    targets[range(targets.shape[0]),y] = 1


    x_sigmoid = np.zeros(l_out.shape[0], np.float32)
    x_sigmoid_hl = []
    for i in range(0,nn.h_layers):
        x_sigmoid_hl.append(np.zeros(l_in.shape[1], np.float32))
    # forward pass
    for i in range(0,len(x)):
        #if i%50000 == 0:
        #    print('i = {}'.format(i))
        x_i=x[i]
        y_i=nn.desired_array_out(y[i])
        if dbg:
            print('l1 shape = {}'.format(l_in.shape))
            #print(l1.shape[0])
            #print(l1.shape[1])
            print('l2 shape = {}'.format(l_out.shape))
            #print(l2.shape[0])
            #print(l2.shape[1])
            #print(x_i)
        #for layer_i in range(0,l1.shape[0]):
            #neuron = l1[layer_i]
        x_l_in = np.dot(l_in.T,np.array(x_i).flatten())
        if dbg:
            print('x_l_in shape={}'.format(x_l_in.shape))
            print('x_l_in={}'.format(x_l_in))
        x_in_sigmoid = sigmoid(x_l_in)
        if dbg:
            print('x_in_sigmoid shape={}'.format(x_in_sigmoid.shape))
            print('x_in_sigmoid={}'.format(x_in_sigmoid))
        for hl_index in range(0, nn.h_layers):
            #print('hl_index=',str(hl_index))
            if hl_index == 0:
                x_sigmoid_hl[hl_index] = hl[hl_index].T@x_in_sigmoid
            else:
                x_sigmoid_hl[hl_index] = hl[hl_index].T @ x_sigmoid_hl[hl_index-1]
        if dbg:
            for hl_index in range(0, nn.h_layers):
                print('x_sigmoid_hl['+str(hl_index)+'] shape={}'.format(x_sigmoid_hl[hl_index].shape))
                print('x_sigmoid_hl['+str(hl_index)+']={}'.format(x_sigmoid_hl[hl_index]))
        if dbg:
            print('x_sigmoid shape={}'.format(x_sigmoid.shape))

        x_l_out = np.dot(x_sigmoid_hl[hl_index].T, l_out)
        if dbg:
            print('x_l_out shape={}'.format(x_l_out.shape))
            print('x_l_out={}'.format(x_l_out))
        out = softmax(x_l_out)
        #out = sigmoid(x_l_out)


        if dbg:
            print('out shape={}'.format(out.shape))
            print('out={}'.format(out))
        print(out, y_i)
        error = np.power(y_i-out,2).mean()
        if i%1==0:
            print('error at step {:5d}: {:10.6e}'.format(i,error))
        #backward pass

        #print('error={}'.format(error))
        #delta_out = error * d_softmax(x_l_out)
        delta_out = 2.0*error * d_sigmoid(x_l_out)
        #delta_out = error.dot(l_out.T) * d_sigmoid(hl[nn.h_layers-1])
        if dbg:
            print('delta2={}'.format(delta_out))
            print('before updation l2 sum={}'.format(np.sum(l_out)))
            #print('x_sigmoid shape={}'.format(x_sigmoid.shape))
            print('delta2.T shape={}'.format(delta_out.T.shape))
            #print(delta2[:,None])
            #print(x_sigmoid[:,None])
            print('l_out={} before updation'.format(l_out))
        l_out = np.add(l_out, -1. * np.outer(x_sigmoid_hl[nn.h_layers - 1], lr * delta_out))  # delta2.T * x_sigmoid)
        #l_out -= lr*hl[nn.h_layers-1].T.dot(delta_out)#add(l_out, -1.*np.outer(x_sigmoid_hl[nn.h_layers-1],lr*delta_out))#delta2.T * x_sigmoid)
        if dbg:
            print('l_out={} after updation'.format(l_out))
            print('after updation l2 sum={}'.format(np.sum(l_out)))

            #delta1 = ((l2[:, neuron_i]).dot(error.T)).T * d_sigmoid(x_l1)
            print('d_sigmoid(x_l1) shape={}'.format(d_sigmoid(x_l_in).shape))
        for hl_index in reversed(range(0,nn.h_layers)):
            if hl_index == nn.h_layers-1:
                delta_hl = ((l_out).dot(delta_out.T)).T * d_sigmoid(x_sigmoid_hl[hl_index])
                hl[hl_index] = np.add(hl[hl_index],-1.*lr*delta_hl)
            else:
                #print(hl[hl_index+1].shape)
                delta_hl = ((hl[hl_index+1]).dot(delta_hl.T)).T * d_sigmoid(x_sigmoid_hl[hl_index])
                hl[hl_index] = np.add(hl[hl_index], -1.*lr * delta_hl)
        #delta_in = ((l_out).dot(error.T)).T * d_sigmoid(x_l_in)
        delta_in = ((hl[0]).dot(delta_hl.T)).T * d_sigmoid(x_l_in)
        if dbg:
            print('delta1 shape={}'.format(delta_in.shape))
            print('x_i.T shape={}'.format((np.array(x_i).flatten()).T.shape))
        #l1 = np.add(l1, x.T @ delta1)
        #l1 = np.add(l1, (np.array(x_i).flatten()).T@delta1)
        l_in = np.add(l_in, np.outer(np.array(x_i).flatten(),-1.*lr*delta_in))
        #exit(0)
        #print('l_out=',l_out)
        #exit(0)
    return out,l_in,hl,l_out

def predict(x,y,nn,l_in,hl,l_out,y_test,y_pred):
    targets = np.zeros((len(y),10), np.float32)
    targets[range(targets.shape[0]),y] = 1


    x_sigmoid = np.zeros(l_out.shape[0], np.float32)

    x_sigmoid = np.zeros(l_out.shape[0], np.float32)
    x_sigmoid_hl = []
    for i in range(0,nn.h_layers):
        x_sigmoid_hl.append(np.zeros(l_in.shape[1], np.float32))
    # forward pass
    for i in range(0, len(x)):
        # if i%50000 == 0:
        #    print('i = {}'.format(i))
        x_i = x[i]
        y_test.append(y[i])
        y_i = nn.desired_array_out(y[i])
        if dbg:
            print('l1 shape = {}'.format(l_in.shape))
            # print(l1.shape[0])
            # print(l1.shape[1])
            print('l2 shape = {}'.format(l_out.shape))
            # print(l2.shape[0])
            # print(l2.shape[1])
            # print(x_i)
        # for layer_i in range(0,l1.shape[0]):
        # neuron = l1[layer_i]
        x_l_in = np.dot(l_in.T, np.array(x_i).flatten())
        if dbg:
            print('x_l1 shape={}'.format(x_l_in.shape))
        x_in_sigmoid = sigmoid(x_l_in)
        for hl_index in range(0, nn.h_layers):
            # print('hl_index=',str(hl_index))
            if hl_index == 0:
                x_sigmoid_hl[hl_index] = hl[hl_index].T @ x_in_sigmoid
            else:
                x_sigmoid_hl[hl_index] = hl[hl_index].T @ x_sigmoid_hl[hl_index - 1]
        if dbg:
            print('x_sigmoid shape={}'.format(x_sigmoid.shape))

        x_l_out = np.dot(x_sigmoid_hl[hl_index].T, l_out)
        if dbg:
            print('x_l_out shape={}'.format(x_l_out.shape))
        out = softmax(x_l_out)
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
    # dataloader = MNIST_Dataloader()
    # dataloader.show_images(5, 5)
    # dataloader.simple_show()

    nn = NeuralNetwork()
    dataloader = MNIST_Dataloader()
    x_train, y_train = dataloader.get_train_data()
    for i in range(0,len(x_train)):
        for j in range(0,len(x_train[i])):
            x_train[i][j] = x_train[i][j]/255.
    #print(x_train[0])
    #    print(x_train[i][0]/255.)
    #x_train[0] = x_train[0]/255. #normalize data

    #l1, _ = get_l1(x_train,y_train,0)
    #x,y = get_input(x_train,y_train,0)
    #out,update_l1,update_l2=forward_backward_pass(x_train, y_train, nn)
    #print('out = {}'.format(out))

    #print(nn.desired_array_out([3]))
    #print(nn.desired_array_out([9]))
    l_in = nn.layers[0]
    # print(l1)
    hl = nn.layers[1]
    l_out = nn.layers[2]

    print('training started...')
    total_epochs = 1
    for epoch in range(0,total_epochs):
        print('epoch={}'.format(epoch))
        out,l_in,hl,l_out = forward_backward_pass(x_train,y_train,nn,l_in,hl,l_out)
        #print('l_out={}'.format(l_out))

    y_test = []
    y_pred = []


    print('validation started...')
    predict(x_train,y_train,nn,l_in,hl,l_out,y_test,y_pred)
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
