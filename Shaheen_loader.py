import os
import struct
from array import array
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class MNIST_Dataloader:
    def __init__(self):
        self.training_labels_path = "./data/train-labels-idx1-ubyte"
        self.training_images_path = "./data/train-images-idx3-ubyte"
        self.test_labels_path = "data/t10k-labels-idx1-ubyte"
        self.test_images_path = "data/t10k-images-idx3-ubyte"

    def read_image_labels(self, images_path, labels_path):
        labels = []
        images = []
        with open(labels_path, 'rb') as labels_file:
            magic, size = struct.unpack(">II", labels_file.read(8))
            labels_data = array("B", labels_file.read())

        with open(images_path, 'rb') as images_file:
            magic, size, rows, cols = struct.unpack(">IIII", images_file.read(16))
            image_data = array("B", images_file.read())

        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            image = np.array(image_data[i * rows * cols: (i + 1) * rows * cols])
            image = image.reshape(28, 28)
            images[i][:] = image

        return images, labels_data

    def get_test_data(self):
        x_test, y_test = self.read_image_labels(self.test_images_path, self.test_labels_path)
        return x_test, y_test

    def get_train_data(self):
        x_train, y_train = self.read_image_labels(self.training_images_path, self.training_labels_path)
        return x_train, y_train

    def convert_data(self, stat = 'train'):
        if stat =='train':
            images, titles = self.get_train_data()
        elif stat =='test':
            images, titles = self.get_test_data()
        else:
            print('the status should be either train or test')
            return
        x_train = []
        xxx= 0
        for img in range(len(images)):
            x_train.append([])
            for row in range(len(images[0])):
                x_train[img].append([])
                for col in range(len(images[0][0])):
                    if images[img][row][col]==0:
                        x_train[img][row].append(0)
                    else:
                        x_train[img][row].append(1)

        return x_train


    def simple_show(self):
        images, _ = self.get_train_data()
        for i in range(9):
            plt.subplot(330+1  + i)
            plt.imshow(images[i], cmap=plt.get_cmap('gray'))
        plt.show()

    def show_images(self, rows, cols):
        images, titles = self.get_train_data()

        index = 1
        for x in range(rows*cols):
            image = images[x]
            title = titles[x]
            plt.subplot(rows, cols,index)
            plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
            plt.imshow(images[x], cmap=plt.get_cmap('gray'))
            plt.title(title)
            index +=1

        plt.show()

    def show_as_picture (self, sample):
        images, titles = self.get_train_data()
        matr = images[sample]
        pic = np.array(matr,dtype = np.uint8)
        cv.imshow('Sample', pic)
        cv.waitKey(0)


def main():
    dataloader = MNIST_Dataloader()
    #dataloader.show_images(4, 6)
    #dataloader.simple_show()
    #dataloader.show_as_picture (120)


md =MNIST_Dataloader()


tt =md.convert_data(stat='test')

tx = tt[3349]


tr = np.array(tx,dtype = np.uint8)
print(tr)
#cv.imshow('Sample', tr)
#cv.waitKey(0)