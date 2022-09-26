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


def display_images(fname, dogs_imgs=None, cats_imgs=None, rows=3, offset=0):
    '''
    Despliega conjunto de imágenes izquierda y derecha junto a la disparidad
    '''
    # plt.figure(figsize=(20,rows*2.5))
    fig, ax = plt.subplots(rows, 2, figsize=(8, rows * 2.5))
    for i in range(rows):
        ax[i, 0].imshow((dogs_imgs[i + offset] + 1) / 2)
        ax[i, 0].set_title('Left')
        ax[i, 1].imshow((cats_imgs[i + offset] + 1) / 2)
        ax[i, 1].set_title('Right')

    plt.tight_layout()
    #plt.show()
    plt.savefig(fname)

display_images("figure_1.png", dogs_imgs, cats_imgs, rows=3)

# Dataset's configuration
idx = int(BUFFER_SIZE*.8)

train_dogs = tf.data.Dataset.list_files(train_dogs_files[:idx], shuffle=False)
train_cats = tf.data.Dataset.list_files(train_cats_files[:idx], shuffle=False)

test_dogs = tf.data.Dataset.list_files(train_dogs_files[idx:], shuffle=False)
test_cats = tf.data.Dataset.list_files(train_cats_files[idx:], shuffle=False)

train_dataset = tf.data.Dataset.zip((train_dogs, train_cats))
train_dataset = train_dataset.shuffle(buffer_size=idx, reshuffle_each_iteration=True)
train_dataset = train_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.zip((test_dogs, test_cats))
test_dataset = test_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

sample_dog, sample_cat = next(iter(train_dataset))
print(f"Sample image shape {sample_dog[0].shape}")

# Pix2Pix Model
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

generator_g.summary()
generator_f.summary()
discriminator_x.summary()
discriminator_y.summary()

to_cat = generator_g(sample_dog)
to_dog = generator_f(sample_cat)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_dog, to_cat, sample_cat, to_dog]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.savefig("figure_2.png")

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real zebra?')
plt.imshow(discriminator_y(sample_cat)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_x(sample_dog)[0, ..., -1], cmap='RdBu_r')

plt.savefig("figure_3.png")

# Loss
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

# Optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)