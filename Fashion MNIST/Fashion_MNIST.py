
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf




def main():


    dataset,metadata=tfds.load('fashion_mnist',as_supervised=True,with_info=True)
    train_dataset, test_dataset=dataset['train'],dataset['test']
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']












if __name__ == '__main__':
     main()
