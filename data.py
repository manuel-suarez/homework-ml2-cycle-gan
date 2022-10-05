import os
import numpy as np
import tensorflow as tf
from glob import glob
from matplotlib import pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256

AUTOTUNE = tf.data.AUTOTUNE
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
  image = read_and_decode(image)
  image = random_jitter(image)
  image = normalize(image)
  return image

def load_images(img1, img2):
    return preprocess_image_train(img1), preprocess_image_train(img2)

def preprocess_image_test(image):
  image = normalize(image)
  return image

# Dataset's configuration
# train_dataset = tf.data.Dataset.zip((dog_dataset, cat_dataset))
# train_dataset = train_dataset.shuffle(buffer_size=n_images, reshuffle_each_iteration=True)
# train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.batch(BATCH_SIZE).repeat()

# Example configuration
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

def build_data(data_folder, global_batch_size):
    # Input pipeline
    dog_files = np.array(glob(os.path.join(data_folder, 'dog.*.jpg')))
    cat_files = np.array(glob(os.path.join(data_folder, 'cat.*.jpg')))

    BUFFER_SIZE = len(dog_files)

    train_dogs = tf.data.Dataset.list_files(dog_files, shuffle=False)
    #train_dogs = train_dogs.map(preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE).batch(global_batch_size)
    #train_dogs = train_dogs.cache().map(
    #    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    #    BUFFER_SIZE).batch(global_batch_size)

    train_cats = tf.data.Dataset.list_files(cat_files, shuffle=False)
    train_dataset = tf.data.Dataset.zip((train_dogs, train_cats)).map(load_images, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(global_batch_size)
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

    #train_cats = train_cats.map(preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE).batch(global_batch_size)
    #train_cats = train_cats.cache().map(
    #    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    #    BUFFER_SIZE).batch(global_batch_size)

    return train_dataset, BUFFER_SIZE

def generate_figure1(train_dataset):
    sample_dog, sample_cat = next(iter(train_dataset))

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

    return sample_dog, sample_cat

def generate_figure2(epoch, sample_dog, sample_cat, generator_g, generator_f, discriminator_x, discriminator_y):
    to_cat = generator_g(sample_dog)
    to_dog = generator_f(sample_cat)
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
    plt.savefig(f"figure_generator_epoch_{epoch}.png")

    plt.figure(figsize=(8, 8))

    plt.subplot(121)
    plt.title('Is a real cat?')
    plt.imshow(discriminator_y(sample_cat)[0, ..., -1], cmap='RdBu_r')

    plt.subplot(122)
    plt.title('Is a real dog?')
    plt.imshow(discriminator_x(sample_dog)[0, ..., -1], cmap='RdBu_r')

    plt.savefig(f"figure_discriminator_epoch_{epoch}.png")