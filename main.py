# Setup the pipeline
import numpy as np
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
print("TensorFlow version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Especificamos nivel de logging para verificar la estrategia distribuida
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

from glob import glob
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt


AUTOTUNE = tf.data.AUTOTUNE

# Execution strategy
mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices for distributed strategy: {}'.format(mirrored_strategy.num_replicas_in_sync))
# Check GPU execution

# Input pipeline
DATA_FOLDER   = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats/train'
dog_files = np.array(glob(os.path.join(DATA_FOLDER, 'dog.*.jpg')))
cat_files = np.array(glob(os.path.join(DATA_FOLDER, 'cat.*.jpg')))

BUFFER_SIZE = len(dog_files)
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_DIM = 3

INPUT_DIM = (IMG_WIDTH, IMG_HEIGHT, OUTPUT_DIM)
LATENT_DIM = 150
LAMBDA = 10
R_LOSS_FACTOR = 10000

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

EPOCHS = 10

n_images        = dog_files.shape[0]
steps_per_epoch = n_images//BATCH_SIZE_PER_REPLICA
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
    BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

train_cats = tf.data.Dataset.list_files(cat_files, shuffle=False)
train_cats = train_cats.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_cats = train_cats.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

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

# Loss Functions
with mirrored_strategy.scope():
    # Set reduction to `NONE` so you can do the reduction afterwards and divide by global batch size.

    # VAE Loss
    vae_g_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    vae_f_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    # Cycle-GAN Loss
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.NONE)
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    def discriminator_loss(loss_obj, real, generated):
        real_loss = loss_obj(tf.ones_like(real), real)
        generated_loss = loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        per_example_loss = total_disc_loss * 0.5
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    def generator_loss(loss_obj, generated):
        per_example_loss = loss_obj(tf.ones_like(generated), generated)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    def calc_cycle_loss(real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return LAMBDA * loss1
        # return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    def identity_loss(real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss
        # return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

# Metrics
with mirrored_strategy.scope():
    # VAE Metrics
    vae_g_total_loss_tracker = tf.keras.metrics.Mean(name="vae_g_total_loss")
    vae_g_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="vae_g_reconstruction_loss")
    vae_g_kl_loss_tracker = tf.keras.metrics.Mean(name="vae_g_kl_loss")
    vae_f_total_loss_tracker = tf.keras.metrics.Mean(name="vae_f_total_loss")
    vae_f_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="vae_f_reconstruction_loss")
    vae_f_kl_loss_tracker = tf.keras.metrics.Mean(name="vae_f_kl_loss")
    # Cycle-GAN Metrics
    total_cycle_loss_tracker = tf.keras.metrics.Mean(name="total_cycle_loss")
    total_gen_g_loss_tracker = tf.keras.metrics.Mean(name="total_gen_g_loss")
    total_gen_f_loss_tracker = tf.keras.metrics.Mean(name="total_gen_f_loss")
    disc_x_loss_tracker = tf.keras.metrics.Mean(name="disc_x_loss")
    disc_y_loss_tracker = tf.keras.metrics.Mean(name="disc_y_loss")

# A model, an optimizer, and a checkpoint must be created under `strategy.scope`.
# Model, optimizer, checkpoint
with mirrored_strategy.scope():
    # VAE Model
    vae_g = VAE(r_loss_factor=R_LOSS_FACTOR, summary=False)
    vae_f = VAE(r_loss_factor=R_LOSS_FACTOR, summary=False)

    # VAE Optimizers
    vae_g_optimizer = keras.optimizers.Adam()
    vae_f_optimizer = keras.optimizers.Adam()

    # Cycle-GAN Architecture
    generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    
    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
    
    # Optimizers
    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

# Define a train_step without @tf.function
def train_step(data):
    # real_x: dog image
    # real_y: cat image
    real_x, real_y = data

    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # VAE Operations
        
        # VAE generates intermediate representation of outputs
        # VAE G generates X -> X'
        vae_g_x = vae_g.encoder_model(real_x)
        vae_g_z, vae_g_z_mean, vae_g_z_log_var = vae_g.sampler_model(vae_g_x)
        vae_g_y = vae_g.decoder_model(vae_g_z)
        # VAE F generates Y -> Y'
        vae_f_y = vae_f.encoder_model(real_y)
        vae_f_z, vae_f_z_mean, vae_f_z_log_var = vae_f.sampler_model(vae_f_y)
        vae_f_x = vae_f.decoder_model(vae_f_z)

        # VAE G Loss
        vae_g_r_loss = vae_g.r_loss_factor * vae_g_loss(real_y, vae_g_y)
        vae_g_kl_loss = -0.5 * (1 + vae_g_z_log_var - tf.square(vae_g_z_mean) - tf.exp(vae_g_z_log_var))
        vae_g_kl_loss = tf.reduce_mean(tf.reduce_sum(vae_g_kl_loss, axis=1))
        vae_g_total_loss = vae_g_r_loss + vae_g_kl_loss
        # VAE F Loss
        vae_f_r_loss = vae_f.r_loss_factor * vae_f_loss(real_x, vae_f_x)
        vae_f_kl_loss = -0.5 * (1 + vae_f_z_log_var - tf.square(vae_f_z_mean) - tf.exp(vae_f_z_log_var))
        vae_f_kl_loss = tf.reduce_mean(tf.reduce_sum(vae_f_kl_loss, axis=1))
        vae_f_total_loss = vae_f_r_loss + vae_f_kl_loss

        # Cycle-GAN Operations
        # Generator G translates X' -> Y
        # Generator F translates Y' -> X.
        # Here we use the intermediate representation of cats and dogs generated by the VAE
        fake_y = generator_g(vae_g_y, training=True) # vae_g_y instead of real_x
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(vae_f_x, training=True) # vae_f_x instead of real_y
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(loss_obj, disc_fake_y)
        gen_f_loss = generator_loss(loss_obj, disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(loss_obj, disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(loss_obj, disc_real_y, disc_fake_y)

    # VAE Gradients
    vae_g_gradients = tape.gradient(vae_g_total_loss, vae_g.trainable_weights)
    vae_f_gradients = tape.gradient(vae_f_total_loss, vae_f.trainable_weights)

    # VAE optimizer step
    vae_g_optimizer.apply_gradients(zip(vae_g_gradients, vae_g.trainable_weights))
    vae_f_optimizer.apply_gradients(zip(vae_f_gradients, vae_f.trainable_weights))

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
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

    # compute progress
    # vae
    vae_g_total_loss_tracker.update_state(vae_g_total_loss)
    vae_g_reconstruction_loss_tracker.update_state(vae_g_r_loss)
    vae_g_kl_loss_tracker.update_state(vae_g_kl_loss)
    vae_f_total_loss_tracker.update_state(vae_f_total_loss)
    vae_f_reconstruction_loss_tracker.update_state(vae_f_r_loss)
    vae_f_kl_loss_tracker.update_state(vae_f_kl_loss)
    # cycle-gan
    total_cycle_loss_tracker.update_state(total_cycle_loss)
    total_gen_g_loss_tracker.update_state(total_gen_g_loss)
    total_gen_f_loss_tracker.update_state(total_gen_f_loss)
    disc_x_loss_tracker.update_state(disc_x_loss)
    disc_y_loss_tracker.update_state(disc_y_loss)
    return {
        "vae_g_loss": vae_g_total_loss_tracker.result(),
        "vae_g_reconstruction_loss": vae_g_reconstruction_loss_tracker.result(),
        "vae_g_kl_loss": vae_g_kl_loss_tracker.result(),
        "vae_f_loss": vae_f_total_loss_tracker.result(),
        "vae_f_reconstruction_loss": vae_f_reconstruction_loss_tracker.result(),
        "vae_f_kl_loss": vae_f_kl_loss_tracker.result(),
        "total_cycle_loss": total_cycle_loss_tracker.result(),
        "total_gen_g_loss": total_gen_g_loss_tracker.result(),
        "total_gen_f_loss": total_gen_f_loss_tracker.result(),
        "disc_x_loss": disc_x_loss_tracker.result(),
        "disc_y_loss": disc_y_loss_tracker.result()
    }

# `run` replicates the provided computation and runs it with the distributed input.
@tf.function
def distributed_train_step(dist_inputs):
    mirrored_strategy.run(train_step, args=(dist_inputs,))
    # return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

to_cat = generator_g(sample_dog)
to_dog = generator_f(sample_cat)
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
plt.imshow(discriminator_y(sample_cat)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real dog?')
plt.imshow(discriminator_x(sample_dog)[0, ..., -1], cmap='RdBu_r')

plt.savefig('figure_4.png')
print("Model builded")

# Checkpoints
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import TerminateOnNaN
# filepath = 'best_weight_model.h5'
# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='loss',
#                              verbose=1,
#                              save_best_only=True,
#                              save_weights_only=True,
#                              mode='min')
# terminate = TerminateOnNaN()
# callbacks = [checkpoint, terminate]

# Train
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

# train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

train_dataset = tf.data.Dataset.zip((train_dogs, train_cats))
dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

for epoch in range(EPOCHS):
  # TRAIN LOOP
  num_batches = 0
  for x in dist_dataset:
    distributed_train_step(x)
    num_batches += 1

  # if epoch % 2 == 0:
  #   checkpoint.save(checkpoint_prefix)

  template = ("Epoch {}, "
              "vae_g_loss: {}, vae_g_reconstruction_loss: {}, vae_g_kl_loss: {}, "
              "vae_f_loss: {}, vae_f_reconstruction_loss: {}, vae_f_kl_loss: {}, "
              "total_cycle_loss: {}, total_gen_g_loss: {}, total_gen_f_loss: {}, "
              "disc_x_loss: {}, disc_y_loss: {}")
  print(template.format(epoch + 1,
                        vae_g_total_loss_tracker.result(),
                        vae_g_reconstruction_loss_tracker.result(),
                        vae_g_kl_loss_tracker.result(),
                        vae_f_total_loss_tracker.result(),
                        vae_f_reconstruction_loss_tracker.result(),
                        vae_f_kl_loss_tracker.result(),
                        total_cycle_loss_tracker.result(),
                        total_gen_g_loss_tracker.result(),
                        total_gen_f_loss_tracker.result(),
                        disc_x_loss_tracker.result(),
                        disc_y_loss_tracker.result()))

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
# for idx, inp in enumerate(train_dogs.take(5)):
#   generate_images(cyclegan.generator_g, inp, f"testdogimage_{idx+1}")

# for idx, inp in enumerate(train_cats.take(5)):
#   generate_images(cyclegan.generator_f, inp, f"testcatimage_{idx+1}")