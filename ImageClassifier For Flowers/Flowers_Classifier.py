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
    download=True
    if download==True: # toggle if needed to download and sort the pictures to folders(first use)
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

    image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
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

    image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=val_dir,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='sparse')

    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plotImages(augmented_images)


    #### Bulding the Model

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)), #16 filters,
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Dropout(0.2), #20 %
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')# 5 classes of flowers

    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])




    ### Train the Model

    epochs = 80

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=int(np.ceil(train_data_gen.n / float(BATCH_SIZE))),
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(val_data_gen.n / float(BATCH_SIZE)))

    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()











if __name__ == '__main__':
    main()