# Setup the pipeline
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

print(tf.__version__)

# Input pipeline
DATA_FOLDER   = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats/train5000'
dog_files = np.array(glob(os.path.join(DATA_FOLDER, 'dog.*.jpg')))
cat_files = np.array(glob(os.path.join(DATA_FOLDER, 'cat.*.jpg')))

BUFFER_SIZE = len(dog_files)
BATCH_SIZE = 32
IMG_WIDTH = 256
IMG_HEIGHT = 256

n_images        = dog_files.shape[0]
steps_per_epoch = n_images//BATCH_SIZE
print('num image files : ', n_images)
print('steps per epoch : ', steps_per_epoch )

def read_and_decode(file):
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    # img = img / 255.0
    # img = tf.image.resize(img, INPUT_DIM[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img

def load_image(file1):
    return read_and_decode(file1)

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image):
  image = normalize(image)
  return image

# Dataset's configuration
# train_dataset = tf.data.Dataset.zip((dog_dataset, cat_dataset))
# train_dataset = train_dataset.shuffle(buffer_size=n_images, reshuffle_each_iteration=True)
# train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.batch(BATCH_SIZE).repeat()

train_dogs = tf.data.Dataset.list_files(dog_files, shuffle=False)
train_dogs = train_dogs.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dogs = train_dogs.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_cats = tf.data.Dataset.list_files(cat_files, shuffle=False)
train_cats = train_cats.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_cats = train_cats.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

sample_dog = next(iter(train_dogs))
sample_cat = next(iter(train_cats))

plt.subplot(121)
plt.title('Dog')
plt.imshow(sample_dog[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Dog with random jitter')
plt.imshow(random_jitter(sample_dog[0]) * 0.5 + 0.5)

print("Plotting dog")
plt.savefig('figure_1.png')

plt.subplot(121)
plt.title('Cat')
plt.imshow(sample_cat[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Cat with random jitter')
plt.imshow(random_jitter(sample_cat[0]) * 0.5 + 0.5)

print("Plotting cat")
plt.savefig('figure_2.png')

# Configure Pix2Pix model
OUTPUT_CHANNELS = 3

# VAE components
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
        encoder_input = keras.layers.Input(shape=self.input_dim, name='encoder')
        x = encoder_input

        for i in range(self.n_layers_encoder):
            x = keras.layers.Conv2D(filters=self.encoder_conv_filters[i],
                       kernel_size=self.encoder_conv_kernel_size[i],
                       strides=self.encoder_conv_strides[i],
                       padding='same',
                       name='encoder_conv_' + str(i), )(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)
            if self.use_dropout:
                x = keras.layers.Dropout(rate=0.25)(x)

        self.last_conv_size = x.shape[1:]
        x = keras.layers.Flatten()(x)
        encoder_output = keras.layers.Dense(self.output_dim)(x)
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
        decoder_input = keras.layers.Input(shape=self.input_dim, name='decoder')
        x = keras.layers.Dense(np.prod(self.input_conv_dim))(decoder_input)
        x = keras.layers.Reshape(self.input_conv_dim)(x)

        for i in range(self.n_layers_decoder):
            x = keras.layers.Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                                kernel_size=self.decoder_conv_t_kernel_size[i],
                                strides=self.decoder_conv_t_strides[i],
                                padding='same',
                                name='decoder_conv_t_' + str(i))(x)
            if i < self.n_layers_decoder - 1:

                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.LeakyReLU()(x)
                if self.use_dropout:
                    x = keras.layers.Dropout(rate=0.25)(x)
            else:
                x = keras.layers.Activation('sigmoid')(x)
        decoder_output = x
        model = keras.Model(decoder_input, decoder_output)
        return model

    def call(self, inputs):
        '''
        '''
        return self.model(inputs)


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
        input_data = keras.layers.Input(shape=self.latent_dim)
        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(input_data)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(input_data)

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

class VAE(keras.Model):
    def __init__(self, r_loss_factor=1, summary=False, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.r_loss_factor = r_loss_factor

        # Architecture
        self.input_dim = INPUT_DIM
        self.latent_dim = LATENT_DIM
        self.encoder_conv_filters = [64, 64, 64, 64]
        self.encoder_conv_kernel_size = [3, 3, 3, 3]
        self.encoder_conv_strides = [2, 2, 2, 2]
        self.n_layers_encoder = len(self.encoder_conv_filters)

        self.decoder_conv_t_filters = [64, 64, 64, 3]
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
        dog_img = data[0]
        cat_img = data[1]
        with tf.GradientTape() as tape:
            # predict
            x = self.encoder_model(dog_img)
            z, z_mean, z_log_var = self.sampler_model(x)
            pred = self.decoder_model(z)

            # loss
            r_loss = self.r_loss_factor * self.mae(cat_img, pred)
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

# Loss function
# Loss Functions
def discriminator_loss(loss_obj, real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def generator_loss(loss_obj, generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

INPUT_DIM     = (IMG_WIDTH,IMG_HEIGHT,3)
LATENT_DIM    = 150
LAMBDA = 10
R_LOSS_FACTOR = 10000

class CycleGAN(keras.Model):
    def __init__(self, p_lambda=LAMBDA, r_loss_factor=R_LOSS_FACTOR, **kwargs):
        super(CycleGAN, self).__init__(**kwargs)
        self.p_lambda = p_lambda

        # VAE Model
        self.vae_g = VAE(r_loss_factor=r_loss_factor, summary=False)
        self.vae_f = VAE(r_loss_factor=r_loss_factor, summary=False)

        # VAE Optimizers
        self.vae_g_optimizer = keras.optimizers.Adam()
        self.vae_f_optimizer = keras.optimizers.Adam()

        # VAE Loss
        self.vae_g_loss = tf.keras.losses.MeanAbsoluteError()
        self.vae_f_loss = tf.keras.losses.MeanAbsoluteError()

        # VAE Metrics
        self.vae_g_total_loss_tracker = tf.keras.metrics.Mean(name="vae_g_total_loss")
        self.vae_g_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="vae_g_reconstruction_loss")
        self.vae_g_kl_loss_tracker = tf.keras.metrics.Mean(name="vae_g_kl_loss")
        self.vae_f_total_loss_tracker = tf.keras.metrics.Mean(name="vae_f_total_loss")
        self.vae_f_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="vae_f_reconstruction_loss")
        self.vae_f_kl_loss_tracker = tf.keras.metrics.Mean(name="vae_f_kl_loss")

        # Cycle-GAN Architecture
        self.generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        self.generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

        # Optimizers
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Loss
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Cycle-GAN Metrics
        self.total_cycle_loss_tracker = tf.keras.metrics.Mean(name="total_cycle_loss")
        self.total_gen_g_loss_tracker = tf.keras.metrics.Mean(name="total_gen_g_loss")
        self.total_gen_f_loss_tracker = tf.keras.metrics.Mean(name="total_gen_f_loss")
        self.disc_x_loss_tracker = tf.keras.metrics.Mean(name="disc_x_loss")
        self.disc_y_loss_tracker = tf.keras.metrics.Mean(name="disc_y_loss")

        self.training_step = 0
        self.built = True

    @tf.function
    def train_step(self, data):
        self.training_step += 1
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        # real_x: dog image
        # real_y: cat image
        real_x, real_y = data
        with tf.GradientTape(persistent=True) as tape:
            # VAE Operations
            # VAE generates intermediate representation of outpus
            # VAE G generates X -> X'
            vae_g_x = self.vae_g.encoder_model(real_x)
            vae_g_z, vae_g_z_mean, vae_g_z_log_var = self.vae_g.sampler_model(vae_g_x)
            vae_g_y = self.vae_g.decoder_model(vae_g_z)
            # VAE F generates Y -> Y'
            vae_f_y = self.vae_f.encoder_model(real_y)
            vae_f_z, vae_f_z_mean, vae_f_z_log_var = self.vae_f.sampler_model(vae_f_y)
            vae_f_x = self.vae_f.decoder_model(vae_f_z)

            # VAE G Loss
            vae_g_r_loss = self.vae_g.r_loss_factor * self.vae_g.mae(real_y, vae_g_y)
            vae_g_kl_loss = -0.5 * (1 + vae_g_z_log_var - tf.square(vae_g_z_mean) - tf.exp(vae_g_z_log_var))
            vae_g_kl_loss = tf.reduce_mean(tf.reduce_sum(vae_g_kl_loss, axis=1))
            vae_g_total_loss = vae_g_r_loss + vae_g_kl_loss
            # VAE F Loss
            vae_f_r_loss = self.vae_f.r_loss_factor * self.vae_f.mae(real_x, vae_f_x)
            vae_f_kl_loss = -0.5 * (1 + vae_f_z_log_var - tf.square(vae_f_z_mean) - tf.exp(vae_f_z_log_var))
            vae_f_kl_loss = tf.reduce_mean(tf.reduce_sum(vae_f_kl_loss, axis=1))
            vae_f_total_loss = vae_f_r_loss + vae_f_kl_loss

            # Cycle-GAN Operations
            # Generator G translates X' -> Y
            # Generator F translates Y' -> X.
            # Here we use the intermediate representación of cats and dogs generated by the VAE
            fake_y = self.generator_g(vae_g_y, training=True) # vae_g_y instead of real_x
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(vae_f_x, training=True) # vae_f_x instead of real_y
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(self.loss_obj, disc_fake_y)
            gen_f_loss = generator_loss(self.loss_obj, disc_fake_x)

            total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(self.loss_obj, disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(self.loss_obj, disc_real_y, disc_fake_y)

        # VAE Gradients
        vae_g_gradients = tape.gradient(vae_g_total_loss, self.vae_g.trainable_weights)
        vae_f_gradients = tape.gradient(vae_f_total_loss, self.vae_f.trainable_weights)

        # VAE optimizer step
        self.vae_g_optimizer.apply_gradients(zip(vae_g_gradients,
                                                 self.vae_g.trainable_weights))
        self.vae_f_optimizer.apply_gradients(zip(vae_f_gradients,
                                                 self.vae_f.trainable_weights))

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                  self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                  self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                      self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                      self.discriminator_y.trainable_variables))

        # compute progress
        # vae
        self.vae_g_total_loss_tracker.update_state(vae_g_total_loss)
        self.vae_g_reconstruction_loss_tracker.update_state(vae_g_r_loss)
        self.vae_g_kl_loss_tracker.update_state(vae_g_kl_loss)
        self.vae_f_total_loss_tracker.update_state(vae_f_total_loss)
        self.vae_f_reconstruction_loss_tracker.update_state(vae_f_r_loss)
        self.vae_f_kl_loss_tracker.update_state(vae_f_kl_loss)
        # cycle-gan
        self.total_cycle_loss_tracker.update_state(total_cycle_loss)
        self.total_gen_g_loss_tracker.update_state(total_gen_g_loss)
        self.total_gen_f_loss_tracker.update_state(total_gen_f_loss)
        self.disc_x_loss_tracker.update_state(disc_x_loss)
        self.disc_y_loss_tracker.update_state(disc_y_loss)

        # save figure to see progress
        to_cat = self.generator_g(sample_dog, training=False)
        to_dog = self.generator_f(sample_cat, training=False)
        plt.figure(figsize=(8, 8))
        contrast = 8
        imgs = [sample_dog, to_cat, sample_cat, to_dog]
        title = ['Dog', 'To Cat', 'Cat', 'To Dog']

        for i in range(len(imgs)):
            plt.subplot(2, 2, i + 1)
            plt.title(title[i])
            if i % 2 == 0:
                plt.imshow(imgs[i][0] * 0.5 + 0.5)
            else:
                plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
        plt.savefig(f"training_step_{self.training_step}.png")

        return {
            "vae_g_loss": self.vae_g_total_loss_tracker.result(),
            "vae_g_reconstruction_loss": self.vae_g_reconstruction_loss_tracker.result(),
            "vae_g_kl_loss": self.vae_g_kl_loss_tracker.result(),
            "vae_f_loss": self.vae_f_total_loss_tracker.result(),
            "vae_f_reconstruction_loss": self.vae_f_reconstruction_loss_tracker.result(),
            "vae_f_kl_loss": self.vae_f_kl_loss_tracker.result(),
            "total_cycle_loss": self.total_cycle_loss_tracker.result(),
            "total_gen_g_loss": self.total_gen_g_loss_tracker.result(),
            "total_gen_f_loss": self.total_gen_f_loss_tracker.result(),
            "disc_x_loss": self.disc_x_loss_tracker.result(),
            "disc_y_loss": self.disc_y_loss_tracker.result()
        }

cyclegan = CycleGAN(p_lambda=LAMBDA, r_loss_factor=R_LOSS_FACTOR)
to_cat = cyclegan.generator_g(sample_dog)
to_dog = cyclegan.generator_f(sample_cat)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_dog, to_cat, sample_cat, to_dog]
title = ['Dog', 'To Cat', 'Cat', 'To Dog']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.savefig('figure_3.png')

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real cat?')
plt.imshow(cyclegan.discriminator_y(sample_cat)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real dog?')
plt.imshow(cyclegan.discriminator_x(sample_dog)[0, ..., -1], cmap='RdBu_r')

plt.savefig('figure_4.png')
print("Model builded")

# Checkpoints
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TerminateOnNaN
filepath = 'best_weight_model.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')
terminate = TerminateOnNaN()
callbacks = [checkpoint, terminate]

# Training
EPOCHS = 50

# Train
train_dataset = tf.data.Dataset.zip((train_dogs, train_cats))
cyclegan.compile()
cyclegan.fit(train_dataset,
             batch_size      = BATCH_SIZE,
             epochs          = EPOCHS,
             initial_epoch   = 0,
             steps_per_epoch = steps_per_epoch,
             callbacks       = callbacks)
#cyclegan.save_weights("model_vae_cycle_gan.h5")


def generate_images(model, test_input, figname):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(figname)

# Run the trained model on the test dataset
for idx, inp in enumerate(train_dogs.take(5)):
  generate_images(cyclegan.generator_g, inp, f"testdogimage_{idx+1}")

for idx, inp in enumerate(train_cats.take(5)):
  generate_images(cyclegan.generator_f, inp, f"testcatimage_{idx+1}")