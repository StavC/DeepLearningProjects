import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



def main():

    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True) # my location is C:\Users\Stav\.keras\datasets\cats_and_dogs_filtered
    zip_dir_base = os.path.dirname(zip_dir)
    base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)

    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)



    ### Setting Model Parameters

    BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
    IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_SHAPE, IMG_SHAPE),  #  rescaling to (150,150)
                                                               class_mode='binary')# because we have only dogs vs cats

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=validation_dir,
                                                                  shuffle=False,
                                                                  target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                                  class_mode='binary')
    ### visualizing Training Images

    sample_training_images, _ = next(train_data_gen)

    # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()

    plotImages(sample_training_images[:5])  # Plot images 0-4



    #### MODEL CREATION















if __name__ == '__main__':
    main()