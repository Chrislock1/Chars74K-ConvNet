"""Script for loading and returning NumPy vectors with images and one-hot labels"""
import glob
import numpy as np
import numpy.random as nprnd
import matplotlib.image as mpim


class Chars74k:
    def __init__(self):
        self.train_size = 4056
        self.test_size = 3056
        self.images, self.labels = self.data_loader()

        class train(object):
            pass

        class test(object):
            pass

        self.train = train
        self.train.images = self.images[:self.train_size-1]
        self.train.labels = self.labels[:self.train_size-1]
        self.test = test
        self.test.images = self.images[self.train_size-1:]
        self.test.labels = self.labels[self.train_size-1:]

    @staticmethod
    def data_loader():
        # Reading data from the folders and return images and labels as one-hot vectors
        character_directories = glob.glob('chars74k-lite/*')
        alphabet_size = len(character_directories)
        data_set = []
        labels = []
        count = 0
        for char in character_directories:
            letters = glob.glob(char+'/*.jpg')
            labels_count = len(letters)
            char_label = np.zeros((labels_count, alphabet_size))
            char_label[np.arange(labels_count), count] = 1
            labels += char_label.tolist()
            count += 1
            data_set += [np.array(mpim.imread(letter)) for letter in letters]
        images = np.array(data_set)
        labels = np.array(labels)

        # Shuffle before return
        for _ in range(10000):
            a = nprnd.randint(7112)
            b = nprnd.randint(7112)
            images[a], labels[a], images[b], labels[b] = images[b], labels[b], images[a], labels[a]
        return images, labels

