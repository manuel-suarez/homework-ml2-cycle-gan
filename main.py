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

for dog_file, cat_file in zip(train_dogs_files[:5], train_cats_files[:5]):
  print(dog_file, cat_file)

BUFFER_SIZE      = len(train_cats_files)
steps_per_epoch  = BUFFER_SIZE // BATCH_SIZE
print('num image files : ', BUFFER_SIZE)
print('steps per epoch : ', steps_per_epoch )

# Funciones para apertura y decodificación de los archivos
def read_and_decode(file):
    '''
    Lee, decodifica y redimensiona la imagen.
    Aplica aumentación
    '''
    # Lectura y decodificación
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    # Normalización
    img = img / 127.5 - 1
    # Redimensionamiento
    img = tf.image.resize(img, INPUT_DIM[:2],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img


def load_images(dog_file, cat_file, flip=True):
    '''
    Lee el conjunto de imágenes de entrada y las redimensiona al tamaño especificado

    Aumentación: Flip horizontal aleatorio, sincronizado
    '''
    dog_img = read_and_decode(dog_file)
    cat_img = read_and_decode(cat_file)
    # Aumentación (el flip debe aplicarse simultáneamente a las 3 imagenes)
    if flip and tf.random.uniform(()) > 0.5:
        dog_img = tf.image.flip_left_right(dog_img)
        cat_img = tf.image.flip_left_right(cat_img)

    return dog_img, cat_img

dogs_imgs = []
cats_imgs = []
# Cargamos 3 imagenes
for i in range(3):
    dog_img, cat_img = load_images(train_dogs_files[i], train_cats_files[i])
    dogs_imgs.append(dog_img)
    cats_imgs.append(cat_img)
# Verificamos la forma de las imagenes cargadas
print(dogs_imgs[0].shape, cats_imgs[0].shape)