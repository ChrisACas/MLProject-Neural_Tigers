import matplotlib.pyplot as plt
import numpy as np
import copy
from MNIST_Dataloader import MNIST_Dataloader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

class NeuralNetwok: 
    def __init__(self, input_size=28*28, output_size=10, h_layers=1, h_neurons_per_layer=256):
        self.input_size = input_size
        self.output_size = output_size
        self.h_layers = h_layers
        self.h_neurons_per_layer = h_neurons_per_layer
        self.layers = self.init_layers(input_size, h_neurons_per_layer, output_size)

    # TODO: implement a programmable amount of hidden layer initialization
    def init_layers(self, input_size, h_neurons_per_layer, output_size):
        '''
        Get layer info and develop weight array 
        initialize random weights for each connection to next layer
            weight array of output size, in array for every input node 
        return these weight arrays for each node as layer
        '''
        layer1 = np.random.uniform(-.1,.1,size=(input_size, h_neurons_per_layer))\
            /np.sqrt(input_size * h_neurons_per_layer)
        
        layer2 = np.random.uniform(-.1,.1,size=(h_neurons_per_layer, output_size))\
            /np.sqrt(h_neurons_per_layer * output_size)
        
        return [layer1, layer2]
    
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
    return 1/(np.exp(-x)+1)    

#derivative of sigmoid
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax
def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

    #forward and backward pass
def mlp_backpropogation(x,y,l1,l2):
    desired_out = np.zeros((len(y),10), np.float32)
    desired_out[range(desired_out.shape[0]),y] = 1

    # forward pass
    ## input layer to hidden layer
    x_l1=x.dot(l1)
    x_sigmoid=sigmoid(x_l1)
    ## hidden layer to output layer
    x_l2=x_sigmoid.dot(l2)
    out=softmax(x_l2)

    # backpropogation l2
    error=2*(out-desired_out)/out.shape[0]*d_softmax(x_l2)
    update_l2=x_sigmoid.T@error

    
    # backpropogation l1
    error=((l2).dot(error.T)).T*d_sigmoid(x_l1)
    update_l1=x.T@error

    return out,update_l1,update_l2

def predict(x, l1, l2):
    # forward pass
    ## input layer to hidden layer
    x_l1=x.dot(l1)
    x_sigmoid=sigmoid(x_l1)
    ## hidden layer to output layer
    x_l2=x_sigmoid.dot(l2)
    out=softmax(x_l2)

    return out

def add_gaussian_noise(x, noise_mean, noise_sigma):
    x_with_noise = copy.deepcopy(x)
    print(" Adding Guassian Noise")
    noise_mean = 0.0
    noise_sigma = 0.05
    for i in range(0, len(x_with_noise)):
        for j in range(0, len(x_with_noise[i])):
            x_with_noise[i][j] =  x_with_noise[i][j] + np.random.normal(noise_mean, noise_sigma)
            x_with_noise[i][j] = [1.0 if ele > 1.0 else ele for ele in x_with_noise[i][j]]
    return x_with_noise
  
def analytics(y_test, y_pred): 
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))

def plot_traintest(plot_title, train_acc, test_acc, epochs):
    plt.title(plot_title)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, test_acc, label="Validation Accuracy")

    plt.legend(loc="upper left")
    plt.show()
    plt.clf()

def main():
    # dataloader = MNIST_Dataloader()
    # dataloader.show_images(5, 5)
    # dataloader.simple_show()

    nn = NeuralNetwok()
    l1 = nn.layers[0]
    l2 = nn.layers[1]
    
    epochs=200
    lr=0.001
    batch=30

    y_pred_list = []
    y_guassian_pred_list = []
    accuracies, val_accuracies = [], []
    epochs_list=[]

    dataloader = MNIST_Dataloader()
    x_train, y_train = dataloader.get_train_data()
    x_test, y_test = dataloader.get_test_data()

    rand=np.arange(60000)
    np.random.shuffle(rand)

    for i in range(epochs):
        sample=np.random.randint(0,x_train.shape[0],size=(batch))

        x=x_train[sample].reshape((-1,28*28))
        y=y_train[sample]
        out,update_l1,update_l2=mlp_backpropogation(x,y,l1,l2)
                  
        l1=l1-lr*update_l1
        l2=l2-lr*update_l2
        
        # every 10 epochs record accuracy 
        if(i%10==0):   
                        
            # prediction function, get highest probability of classification
            y_pred_list = np.argmax(predict(x_test.reshape((-1,28*28)), l1, l2), axis=1)

            classification=np.argmax(out,axis=1)
            training_accuracy=(classification==y).mean()
            accuracies.append(training_accuracy)
            
            val_acc=(y_pred_list==y_test).mean()
            val_accuracies.append(val_acc.item())
    
            epochs_list.append(i)
            
            if(i==(epochs-10)):
                x_test_w_guassian = add_gaussian_noise(x_test, 0.0, 0.15)
                y_guassian_pred_list = np.argmax(predict(x_test_w_guassian.reshape((-1,28*28)), l1, l2), axis=1)
                gaussian_acc=(y_guassian_pred_list==y_test).mean()
                print(f'Epoch {i}: Training Accuracy: {training_accuracy:.3f} | Validation Accuracy w/ Gaussian:{gaussian_acc:.3f}')  

        if(i%10==0): print(f'Epoch {i}: Training Accuracy: {training_accuracy:.3f} | Validation Accuracy:{val_acc:.3f}')

    y_pred = np.array(y_pred_list)
    confusion = confusion_matrix(y_test, y_pred)
    print(confusion)
    
    # Normal Training and Testing Analytics
    analytics(y_test, y_pred)
    plot_title = "Epoch v Accuracy"
    plot_traintest(plot_title, accuracies, val_accuracies, epochs_list)

    # Normal Training and Testing with Guassian Noise Analytics
    print("====================================================")
    print("Analytics of Accuracies when Gaussian noise is added")
    analytics(y_test, y_guassian_pred_list)
   

    
  
if __name__=="__main__":
    main()