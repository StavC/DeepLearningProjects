import os
import matplotlib.pyplot
import tensorflow as tf
import numpy as np



def main():

    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True) # my location is C:\Users\Stav\.keras\datasets\cats_and_dogs_filtered



if __name__ == '__main__':
    main()