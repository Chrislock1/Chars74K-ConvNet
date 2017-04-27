""" Script containing neural network model"""
import tensorflow as tf
import numpy as np
from Chars74K_DataLoader import Chars74k

chars74k = Chars74k()

train_images = chars74k.train.images
train_labels = chars74k.train.labels
test_images = chars74k.test.images
test_labels = chars74k.test.labels

# Build model
x = tf.placeholder(name='Input_Images', dtype=tf.float32, shape=[None, 20, 20, 1])
y = tf.placeholder(name='Target_label', dtype=tf.int32, shape=[None, 26])

