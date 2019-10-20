
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf




def main():

    def normalize(images, labels):### making each pixel in range of [0,1] instead of [0.255]
        images = tf.cast(images, tf.float32)
        images /= 255
        return images, labels

    dataset,metadata=tfds.load('fashion_mnist',as_supervised=True,with_info=True)
    train_dataset, test_dataset=dataset['train'],dataset['test']
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples

    print("Number of training examples: {}".format(num_train_examples))
    print("Number of test examples:     {}".format(num_test_examples))

    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)

    for image, label in test_dataset.take(1):
        break
    image = image.numpy().reshape((28, 28))

    # Plot the image
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    # Plot the first 25 images
    plt.figure(figsize=(10, 10))
    i = 0
    for (image, label) in test_dataset.take(25):
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
        i += 1
    plt.show()


    #### Build the Model

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),#This layer transforms the images from a 2d-array of 28  ×  28 pixels, to a 1d-array of 784 pixels (28*28). Think of this layer as unstacking rows of pixels in the image and lining them up
        tf.keras.layers.Dense(128, activation='relu'),#A densely connected layer of 128 neurons
        tf.keras.layers.Dense(10, activation='softmax')#output tf.keras.layers.Dense — A 10-node softmax layer, with each node representing a class of clothing.
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    BATCH_SIZE = 32
    train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)# Repeat forever but limited by Epochs
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

if __name__ == '__main__':
     main()
