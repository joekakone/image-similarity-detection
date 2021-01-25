# coding : utf-8


import os
import math

import glob
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


def load_transfer_values(path):
    features = np.load(path)

    return features

def load_image(image_path, size):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(size)
    image = np.array(image) / 255.0
    
    return image

def read_data(data_dir):
	classes = [it for it in os.listdir(data_dir) if it != 'LICENSE.txt']
	classes = sorted(classes) # sort list
	classes = {classes[i]: i for i in range(len(classes))}
	num_classes = len(classes)

	# listing all images paths
	filenames = glob.glob(data_dir+'*/*.jpg')
	np.random.shuffle(filenames)
	labels = [path.split('/')[-2] for path in filenames]
	labels = np.array([classes[label] for label in labels])

	return filenames, labels, num_classes

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    outputs = model.layers[-3].output
    model = tf.keras.models.Model(inputs=model.input, outputs=outputs)

    return model

def extract_image_id(image_path):
        image_id = image_path.split("/")[-1]
        image_id = image_id.split(".")
        if len(image_id)>1:
            image_id = ".".join(image_id[:-1])
        else:
            image_id = image_id[0]

        return image_id


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size=32, target_size=(150, 150)):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            load_image(file_name, self.target_size)
               for file_name in batch_x]), np.array(batch_y)
