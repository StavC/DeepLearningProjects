import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import  LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf





def main():
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    dataset=pd.read_csv('BankDataSet.csv')
    inputs=dataset.copy()

    targets=dataset.iloc[:,-1].values # get the targets
    countries = pd.get_dummies(inputs['Geography'], drop_first=True) # dummies
    inputs['Geography']=countries # add the dummies
    inputs=inputs.iloc[:,3:13] # discard useless data
    inputs['Gender'] = inputs['Gender'].map({'Male': 0, 'Female': 1}) #handling the Gender
    print(inputs['Gender'])
    print(targets)
    ###spliting the data
    x_train,x_test,y_train,y_test=train_test_split(inputs,targets,test_size=0.2,random_state=0)
    ########### x is the input for training and testing and y is the output for training and testing

    #### Feature Scaling
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train) # fitting and transforming the inputs for training
    x_test=sc.transform(x_test)   # transofrming the inputs for testing

    #### Make the ANN
    hidden_layer_size=6 #input layer has 11 nodes and in output is 1 so (11+1)/2=6
    output_size = 1 # output is 1
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=hidden_layer_size, kernel_initializer='glorot_uniform',activation='relu')) #adding a Layer
    model.add(tf.keras.layers.Dense(units=hidden_layer_size, kernel_initializer='glorot_uniform',activation='relu')) #adding a Layer
    model.add(tf.keras.layers.Dense(units=output_size, kernel_initializer='glorot_uniform',activation='sigmoid')) #sigmoid is good for 1 if there will be more than two categorize will choose softmax as activation

    #### compiling the ANN
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])






if __name__ == '__main__':
    main()