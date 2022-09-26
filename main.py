# Importación de blbliotecas
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow_examples.models.pix2pix import pix2pix
from keras.utils.vis_utils import plot_model

AUTOTUNE = tf.data.AUTOTUNE

print(tf.__version__)

# Variables de configuración
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

INPUT_DIM     = (256,256,3)
OUTPUT_CHANNELS = 1
BATCH_SIZE    = 10
R_LOSS_FACTOR = 10000
EPOCHS        = 100
INITIAL_EPOCH = 0

# Configuración de rutas para la carga de conjuntos de datos
local_path = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats/train'
train_dogs_files = glob(os.path.join(local_path, 'dog.*.jpg'))
train_cats_files = glob(os.path.join(local_path, 'cat.*.jpg'))
train_dogs_files.sort()
train_cats_files.sort()
train_dogs_files = np.array(train_dogs_files)
train_cats_files = np.array(train_cats_files)

print(len(train_dogs_files), len(train_cats_files))