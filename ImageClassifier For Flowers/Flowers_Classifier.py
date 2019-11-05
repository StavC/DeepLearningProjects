import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf



def main():
    # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()



    _URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    zip_file = tf.keras.utils.get_file(origin=_URL,
                                       fname="flower_photos.tgz",
                                       extract=True)

    base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

    classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

    for cl in classes:
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.jpg')
        print("{}: {} Images".format(cl, len(images)))
        train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]

        for t in train:
            if not os.path.exists(os.path.join(base_dir, 'train', cl)):
                os.makedirs(os.path.join(base_dir, 'train', cl))
            shutil.move(t, os.path.join(base_dir, 'train', cl))

        for v in val:
            if not os.path.exists(os.path.join(base_dir, 'val', cl)):
                os.makedirs(os.path.join(base_dir, 'val', cl))
            shutil.move(v, os.path.join(base_dir, 'val', cl))



        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'val')


        #### Data Augmentation

        BATCH_SIZE = 100
        IMG_SHAPE = 150

        image_gen_train = tf.image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=45,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.5,
            zoom_range=0.5,
            horizontal_flip=True,
            fill_mode='nearest'

        )

        train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                             directory=train_dir,
                                                             shuffle=True,
                                                             target_size=(IMG_SHAPE, IMG_SHAPE),
                                                             class_mode='sparse')

        augmented_images = [train_data_gen[0][0][0] for i in range(5)]
        plotImages(augmented_images)







if __name__ == '__main__':
    main()