import numpy as np # linear algebra
import struct
from array import array
import matplotlib.pyplot as plt

class MNIST_Dataloader: 
    '''
    http://yann.lecun.com/exdb/mnist/
    '''
    def __init__(self): 
        self.training_labels_path = "./data/train-labels-idx1-ubyte"
        self.training_images_path = "./data/train-images-idx3-ubyte"
        self.test_labels_path = "data/t10k-labels-idx1-ubyte"
        self.test_images_path = "data/t10k-images-idx3-ubyte"
        
    '''
    open label file and open image file with their file paths

    return 60k label element array (each element is 0-9 numeric label)
        and 60k image element array (each element 28x28 matrix pixel grid)
    '''
    def read_image_labels(self, labels_path, images_path): 
        labels = []
        images = []

        # open labels file 
        #   extract first 8 bytes with file info
        #   extract remaining bytes to array and turn them to # value from byte
        with open(labels_path, 'rb') as labels_file:
            magic, size = struct.unpack(">II", labels_file.read(8)) 
            labels = array("B", labels_file.read())                 

        # open image file 
        #   extract first 16 bytes with file info
        #   extract rest to simple 1D array of len 60k * 784
        #       each element holds grayscale value
        with open(images_path, 'rb') as images_file:
            magic, size, rows, cols = struct.unpack(">IIII", images_file.read(16))     
            image_data = array("B", images_file.read())
            
        # initialize empty 60k len array with entries of a 784 element arrays
        for i in range(size):
            images.append([0] * rows * cols)

        # get 784 element segments from stream and put in numpy array
        # reshape numpy array to be 28x28 matrix representing pixel grid of img
        # put matrix in previously init'd array
        for i in range(size): 
            image = np.array(image_data[i*rows*cols: (i+1)*rows*cols])
            image = image.reshape(28, 28)
            images[i][:] = image

        return images, labels

    def get_test_data(self):
        x_test, y_test = self.read_image_labels(self.test_labels_path, self.test_images_path,)
        return np.array(x_test), np.array(y_test)

    def get_train_data(self):
        x_train, y_train = self.read_image_labels(self.training_labels_path, self.training_images_path)
        return np.array(x_train), np.array(y_train)


    def get_image_shape(self):
        # TODO: Change this to show either train or test images
        with open(self.training_images_path, 'rb') as images_file:
            magic, size, rows, cols = struct.unpack(">IIII", images_file.read(16))
        return rows, cols

    def show_images(self, rows, cols ): 
        # TODO: Change this to show either train or test
        images, labels = self.get_train_data()
        index = 1
        for x in zip(images, labels):
            image = x[0]
            label = x[1]
            plt.subplot(rows, cols, index)      # map out the number of suplots  
            plt.tight_layout(pad=1)             # pad the image for titles
            plt.imshow(image, cmap="gray")      # set image matrix to grayscale
            if (labels != ''):
                plt.title(label, fontsize = 10) # provide plot for title    
            if index == rows*cols:              # stop adding images when full
                break    
            index += 1

        plt.show()
        
    def simple_show(self):
        images, _ = self.get_train_data()
        for i in range(9):  
            plt.subplot(330+1 + i)
            plt.imshow(images[i], cmap=plt.get_cmap('gray'))
        plt.show()








    