# Setup the pipeline
import tensorflow as tf
import time
# tf.debugging.set_log_device_placement(True)
info = tf.compat.v1.logging.info
# Especificamos nivel de logging para verificar la estrategia distribuida
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
info(f"TensorFlow version: {tf.__version__}")
info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

#configproto = tf.compat.v1.ConfigProto()
#configproto.gpu_options.allow_growth = True
#configproto.gpu_options.polling_inactive_delay_msecs = 10
#sess = tf.compat.v1.Session(config=configproto)
#tf.compat.v1.keras.backend.set_session(sess)

from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow_examples.models.pix2pix.pix2pix import InstanceNormalization

import matplotlib.pyplot as plt

from models import VAE
from data import build_data, generate_figure1, generate_figure2

# Execution strategy
mirrored_strategy = tf.distribute.MirroredStrategy()
info('Number of devices for distributed strategy: {}'.format(mirrored_strategy.num_replicas_in_sync))
# Check GPU execution

BATCH_SIZE_PER_REPLICA = 1
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

DATA_FOLDER = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats/train'
train_dataset, BUFFER_SIZE = build_data(DATA_FOLDER, global_batch_size=GLOBAL_BATCH_SIZE)
sample_dog, sample_cat = generate_figure1(train_dataset)

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_DIM = 3

INPUT_DIM = (IMG_WIDTH, IMG_HEIGHT, OUTPUT_DIM)
LATENT_DIM = 150
LAMBDA = 10
R_LOSS_FACTOR = 10000


EPOCHS = 30

steps_per_epoch = BUFFER_SIZE//BATCH_SIZE_PER_REPLICA
info(f"num image files : {BUFFER_SIZE}")
info(f"steps per epoch : {steps_per_epoch}")

# Configure Pix2Pix model
OUTPUT_CHANNELS = 3

# Loss Functions
info("Loss functions")
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
info("Metrics")
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
info("Model, optimizer")
with mirrored_strategy.scope():
    # VAE Model
    vae_g = VAE(r_loss_factor=R_LOSS_FACTOR, summary=False, input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    vae_f = VAE(r_loss_factor=R_LOSS_FACTOR, summary=False, input_dim=INPUT_DIM, latent_dim=LATENT_DIM)

    # VAE Optimizers
    vae_g_optimizer = keras.optimizers.Adam()
    vae_f_optimizer = keras.optimizers.Adam()

    # Cycle-GAN Architecture
    generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='batchnorm')
    generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='batchnorm')
    
    discriminator_x = pix2pix.discriminator(norm_type='batchnorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='batchnorm', target=False)
    
    # Optimizers
    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

# Define a train_step without @tf.function
info("Train step")
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
info("Distributed train step")
@tf.function
def distributed_train_step(dist_inputs):
    mirrored_strategy.run(train_step, args=(dist_inputs,))
    # return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

generate_figure2(0, sample_dog, sample_cat, generator_g, generator_f, discriminator_x, discriminator_y)
info("Model builded")

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
info("Train dataset")
# train_dataset = tf.data.Dataset.zip((train_dogs, train_cats))
info("Distributed dataset")
dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

info("Training")
for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    # TRAIN LOOP
    for step, data in enumerate(dist_dataset):
      distributed_train_step(data)

      # if epoch % 2 == 0:
      #   checkpoint.save(checkpoint_prefix)

    info("Time taken: %.2fs" % (time.time() - start_time))
    template = ("Epoch {}, "
              "vae_g_loss: {}, vae_g_reconstruction_loss: {}, vae_g_kl_loss: {}, "
              "vae_f_loss: {}, vae_f_reconstruction_loss: {}, vae_f_kl_loss: {}, "
              "total_cycle_loss: {}, total_gen_g_loss: {}, total_gen_f_loss: {}, "
              "disc_x_loss: {}, disc_y_loss: {}")
    info(template.format(epoch + 1,
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

    # Generamos imagen para observar desempeño de la red
    generate_figure2(epoch + 1, sample_dog, sample_cat, generator_g, generator_f, discriminator_x, discriminator_y)

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