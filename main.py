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
BATCH_SIZE = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256

INPUT_DIM     = (256,256,3)
OUTPUT_CHANNELS = 1
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


def load_images(file1, file2, flip=True):
    '''
    Lee el conjunto de imágenes de entrada y las redimensiona al tamaño especificado

    Aumentación: Flip horizontal aleatorio, sincronizado
    '''
    img1 = read_and_decode(file1)
    img2 = read_and_decode(file2)
    # Aumentación (el flip debe aplicarse simultáneamente a las 3 imagenes)
    if flip and tf.random.uniform(()) > 0.5:
        img1 = tf.image.flip_left_right(img1)
        img2 = tf.image.flip_left_right(img2)

    return img1, img2

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
print(sample_dog[0].shape, sample_cat[0].shape)

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
title = ['Dog', 'To cat', 'Cat', 'To dog']

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
plt.title('Is a real cat?')
plt.imshow(discriminator_y(sample_cat)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real dog?')
plt.imshow(discriminator_x(sample_dog)[0, ..., -1], cmap='RdBu_r')

plt.savefig("figure_3.png")

# VAE
class Sampler(keras.Model):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, latent_dim, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.model = self.sampler_model()
        self.built = True

    def get_config(self):
        config = super(Sampler, self).get_config()
        config.update({"units": self.units})
        return config

    def sampler_model(self):
        '''
        input_dim is a vector in the latent (codified) space
        '''
        input_data = layers.Input(shape=self.latent_dim)
        z_mean = Dense(self.latent_dim, name="z_mean")(input_data)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(input_data)

        self.batch = tf.shape(z_mean)[0]
        self.dim = tf.shape(z_mean)[1]

        epsilon = tf.keras.backend.random_normal(shape=(self.batch, self.dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        model = keras.Model(input_data, [z, z_mean, z_log_var])
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)

class Encoder(keras.Model):
    def __init__(self, input_dim, output_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):
        '''
        '''
        super(Encoder, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.n_layers_encoder = len(self.encoder_conv_filters)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.model = self.encoder_model()
        self.built = True

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"units": self.units})
        return config

    def encoder_model(self):
        '''
        '''
        encoder_input = layers.Input(shape=self.input_dim, name='encoder')
        x = encoder_input

        for i in range(self.n_layers_encoder):
            x = Conv2D(filters=self.encoder_conv_filters[i],
                       kernel_size=self.encoder_conv_kernel_size[i],
                       strides=self.encoder_conv_strides[i],
                       padding='same',
                       name='encoder_conv_' + str(i), )(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        self.last_conv_size = x.shape[1:]
        x = Flatten()(x)
        encoder_output = Dense(self.output_dim)(x)
        model = keras.Model(encoder_input, encoder_output)
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)

class Decoder(keras.Model):
    def __init__(self, input_dim, input_conv_dim,
                 decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):

        '''
        '''
        super(Decoder, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.input_conv_dim = input_conv_dim

        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.n_layers_decoder = len(self.decoder_conv_t_filters)

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.model = self.decoder_model()
        self.built = True

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"units": self.units})
        return config

    def decoder_model(self):
        '''
        '''
        decoder_input = layers.Input(shape=self.input_dim, name='decoder')
        x = Dense(np.prod(self.input_conv_dim))(decoder_input)
        x = Reshape(self.input_conv_dim)(x)

        for i in range(self.n_layers_decoder):
            x = Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                                kernel_size=self.decoder_conv_t_kernel_size[i],
                                strides=self.decoder_conv_t_strides[i],
                                padding='same',
                                name='decoder_conv_t_' + str(i))(x)
            if i < self.n_layers_decoder - 1:

                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)
        decoder_output = x
        model = keras.Model(decoder_input, decoder_output)
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)
    # Loss

# Dimensión de la imagen de entrada (el polinomio) utilizado en el entrenamiento y pruebas
INPUT_DIM     = (256,256,OUTPUT_CHANNELS)
# Dimensión del espacio latente
LATENT_DIM    = 150
BATCH_SIZE    = 10
R_LOSS_FACTOR = 100000  # 10000
EPOCHS        = 50
INITIAL_EPOCH = 0
use_batch_norm  = True
use_dropout     = True

class VAE(keras.Model):
    def __init__(self, r_loss_factor=1, summary=False, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.r_loss_factor = r_loss_factor

        # Architecture
        self.input_dim = INPUT_DIM
        self.latent_dim = LATENT_DIM
        # Utilizamos un número mayor de capas convolucionales para obtener mejor
        # las características del gradiente de entrada
        self.encoder_conv_filters = [64, 64, 64, 64]
        self.encoder_conv_kernel_size = [3, 3, 3, 3]
        self.encoder_conv_strides = [2, 2, 2, 2]
        self.n_layers_encoder = len(self.encoder_conv_filters)

        self.decoder_conv_t_filters = [64, 64, 64, OUTPUT_CHANNELS]
        self.decoder_conv_t_kernel_size = [3, 3, 3, 3]
        self.decoder_conv_t_strides = [2, 2, 2, 2]
        self.n_layers_decoder = len(self.decoder_conv_t_filters)

        self.use_batch_norm = True
        self.use_dropout = True

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.mae = tf.keras.losses.MeanAbsoluteError()

        # Encoder
        self.encoder_model = Encoder(input_dim=self.input_dim,
                                     output_dim=self.latent_dim,
                                     encoder_conv_filters=self.encoder_conv_filters,
                                     encoder_conv_kernel_size=self.encoder_conv_kernel_size,
                                     encoder_conv_strides=self.encoder_conv_strides,
                                     use_batch_norm=self.use_batch_norm,
                                     use_dropout=self.use_dropout)
        self.encoder_conv_size = self.encoder_model.last_conv_size
        if summary:
            self.encoder_model.summary()

        # Sampler
        self.sampler_model = Sampler(latent_dim=self.latent_dim)
        if summary:
            self.sampler_model.summary()

        # Decoder
        self.decoder_model = Decoder(input_dim=self.latent_dim,
                                     input_conv_dim=self.encoder_conv_size,
                                     decoder_conv_t_filters=self.decoder_conv_t_filters,
                                     decoder_conv_t_kernel_size=self.decoder_conv_t_kernel_size,
                                     decoder_conv_t_strides=self.decoder_conv_t_strides,
                                     use_batch_norm=self.use_batch_norm,
                                     use_dropout=self.use_dropout)
        if summary: self.decoder_model.summary()

        self.built = True

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker, ]

    @tf.function
    def train_step(self, data):
        print(data)
        '''
        '''
        # Desestructuramos data ya que contiene los dos inputs (gradientes, integral)
        dog, cat = data
        with tf.GradientTape() as tape:
            # predict
            x = self.encoder_model(dog)
            z, z_mean, z_log_var = self.sampler_model(x)
            pred = self.decoder_model(z)

            # loss
            r_loss = self.r_loss_factor * self.mae(cat, pred)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = r_loss + kl_loss

        # gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        # train step
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # compute progress
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(), }

    @tf.function
    def generate(self, z_sample):
        '''
        We use the sample of the N(0,I) directly as
        input of the deterministic generator.
        '''
        return self.decoder_model(z_sample)

    @tf.function
    def codify(self, images):
        '''
        For an input image we obtain its particular distribution:
        its mean, its variance (unvertaintly) and a sample z of such distribution.
        '''
        x = self.encoder_model.predict(images)
        z, z_mean, z_log_var = self.sampler_model(x)
        return z, z_mean, z_log_var

    # implement the call method
    @tf.function
    def call(self, inputs, training=False):
        '''
        '''
        tmp1, tmp2 = self.encoder_model.use_Dropout, self.decoder_model.use_Dropout
        if not training:
            self.encoder_model.use_Dropout, self.decoder_model.use_Dropout = False, False

        x = self.encoder_model(inputs)
        z, z_mean, z_log_var = self.sampler_model(x)
        pred = self.decoder_model(z)

        self.encoder_model.use_Dropout, self.decoder_model.use_Dropout = tmp1, tmp2
        return pred

vae_g = VAE(r_loss_factor=R_LOSS_FACTOR, summary=False)
vae_g.summary()

vae_f = VAE(r_loss_factor=R_LOSS_FACTOR, summary=False)
vae_f.summary()

# LOSS functions
LAMBDA = 10
# CycleGAN
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# VAE
mae      = tf.keras.losses.MeanAbsoluteError()

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

def vae_loss(image, prediction, z_mean, z_log_var, r_loss_factor):
  r_loss     = r_loss_factor * mae(image, prediction)
  kl_loss    = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
  kl_loss    = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
  return r_loss + kl_loss # Total loss

# Optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# VAE
vae_f_optimizer = keras.optimizers.Adam()
vae_g_optimizer = keras.optimizers.Adam()

# Checkpoints
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           vae_f=vae_f,
                           vae_g=vae_g,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer,
                           vae_f_optimizer=vae_f_optimizer,
                           vae_g_optimizer=vae_g_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

# Training
EPOCHS = 10

def generate_images(model, test_input, fname):
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig(fname)

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.

  # Introducimos la variación en el algoritmo con el uso de la VAE
  # X -> Vae g -> X' -> Generator g -> Fake Y
  # Y -> Vae f -> Y' -> Generator f -> Fake X
  with tf.GradientTape(persistent=True) as tape:
    # VAE G generates X -> X' (Dog to Cat)
    sample_g                   = vae_g.encoder_model(real_x)
    z_g, z_mean_g, z_log_var_g = vae_g.sampler_model(sample_g)
    vae_x                      = vae_g.decoder_model(z_g)
    # VAE F generates Y -> Y' (Cat to Dog)
    sample_f                   = vae_f.encoder_model(real_y)
    z_f, z_mean_f, z_log_var_f = vae_f.sampler_model(sample_f)
    vae_y                      = vae_f.decoder_model(z_f)

    # Generator G translates X' -> Y (Cat' to Cat)
    # Generator F translates Y -> X (Cat to Dog)
    # Reemplazamos la entrada real_x (Dog) por la generada por la VAE G (Dog')
    # fake_y = generator_g(real_x, training=True)
    fake_y = generator_g(vae_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    # Reemplazamos la entrada real_y (Cat) por la generada por la VAE F (Cat')
    # fake_x = generator_f(real_y, training=True)
    fake_x = generator_f(vae_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    # VAE Loss
    vae_g_total_loss = vae_loss(real_y, vae_x, z_mean_g, z_log_var_g, vae_g.r_loss_factor)
    vae_f_total_loss = vae_loss(real_x, vae_y, z_mean_f, z_log_var_f, vae_f.r_loss_factor)

    # CycleGAN Loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for VAE
  vae_g_gradients = tape.gradient(vae_g_total_loss, vae_g.trainable_variables)
  vae_f_gradients = tape.gradient(vae_f_total_loss, vae_f.trainable_variables)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss,
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss,
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss,
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss,
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  # VAE
  vae_g_optimizer.apply_gradients(zip(vae_g_gradients,
                                      vae_g.trainable_variables))
  vae_f_optimizer.apply_gradients(zip(vae_f_gradients,
                                      vae_f.trainable_variables))

  # CycleGAN
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

# Train
for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in train_dataset:
    #tf.data.Dataset.zip((train_horses, train_zebras)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, sample_dog, f"generate_{epoch}.png")

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))