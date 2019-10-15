import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

class Model(object):

    def __init__(self):
        return

    def build_model(self):
        #### Make the ANN
        hidden_layer_size = 6  # input layer has 11 nodes and in output is 1 so (11+1)/2=6
        output_size = 1  # output is 1
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=hidden_layer_size, kernel_initializer='glorot_uniform',
                                        activation='relu'))  # adding a Layer
        model.add(tf.keras.layers.Dense(units=hidden_layer_size, kernel_initializer='glorot_uniform',
                                        activation='relu'))  # adding a Layer
        model.add(tf.keras.layers.Dense(units=output_size, kernel_initializer='glorot_uniform',
                                        activation='sigmoid'))  # sigmoid is good for 1 if there will be more than two categorize will choose softmax as activation

        #### compiling the ANN
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model