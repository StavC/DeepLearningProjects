import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import  LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix





def main():
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    dataset=pd.read_csv('BankDataSet.csv')
    inputs=dataset.copy()

    targets=dataset.iloc[:,-1].values # get the targets
    countries = pd.get_dummies(inputs['Geography'], drop_first=True) # dummies
    Spain=countries['Spain'] # add the dummies
    Germany=countries['Germany'] # add the dummies
    inputs=inputs.iloc[:,3:13] # discard useless data
    inputs['Gender'] = inputs['Gender'].map({'Male': 0, 'Female': 1}) #handling the Gender
    inputs=inputs.drop(['Geography'],axis=1)
    inputs=pd.concat([inputs,Spain,Germany],axis=1)
    print(inputs.head())


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

    #### fit the ANN to the training set

    batch_size = 10
    max_epochs = 100
    model.fit(x_train,y_train,batch_size=batch_size,epochs=max_epochs)
    # we got to 86% accuracy

    #### predicting
    prediction=model.predict(x_test)
    prediction=(prediction > 0.5)

    #### making the Confusion Martix

    cm=confusion_matrix(y_test,prediction)
    print(cm)

    print(f" the accuracy is:  {(cm[0][0]+cm[1][1])/cm.sum()}") ## printing accuracy by the CM

    #### another way to test the model
    #test_loss, test_accuracy = model.evaluate(x_test, y_test)
    #print(f'test loss {test_loss}  test accuracy {test_accuracy}')


    ### predicting a single new observation
    single=np.array([[600,1,40,3,60000,2,1,1,50000,0.0,0]])
    single=sc.transform(single)
    new_prediction=model.predict(single)
    print(f" the chances that he will leave the bank are: {float(new_prediction)}")
    print(new_prediction>0.5)










if __name__ == '__main__':
    main()