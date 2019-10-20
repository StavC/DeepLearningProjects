
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from PredictingAppRating import CustomScaler


def main():


    def build_model():
        #### Make the ANN
        hidden_layer_size = 6  # input layer has 11 nodes and in output is 1 so (11+1)/2=6
        output_size = 1  # output is 1
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=hidden_layer_size, kernel_initializer='glorot_uniform',
                                        activation='relu'))  # adding a Layer with dropout
        model.add(tf.keras.layers.Dropout(rate=0.1,))#start with low rate  never get over 0.5
        model.add(tf.keras.layers.Dense(units=hidden_layer_size, kernel_initializer='glorot_uniform',
                                        activation='relu'))  # adding a Layer
        model.add(tf.keras.layers.Dropout(rate=0.1))#start with low rate  never get over 0.5

        model.add(tf.keras.layers.Dense(units=output_size, kernel_initializer='glorot_uniform',
                                        activation='softmax'))  # sigmoid is good for 1 if there will be more than two categorize will choose softmax as activation
        #### compiling the ANN
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    dataset = pd.read_csv('googleplaystore.csv')
    #print(dataset.info())
    dataset.dropna(inplace=True)
    #print(dataset.info())

    category=pd.get_dummies(dataset['Category'],drop_first=True)
    Content_Rating = pd.get_dummies(dataset['Content Rating'], drop_first=True)
    Genres = pd.get_dummies(dataset['Genres'], drop_first=True)
    dataset = dataset.drop(['Category'], axis=1)
    dataset = dataset.drop(['Content Rating'], axis=1)
    dataset = dataset.drop(['Genres'], axis=1)
    dataset=pd.concat([dataset,category,Content_Rating],axis=1)

    #
    #print(dataset.info())
    dataset['Installs'] = dataset['Installs'].apply(lambda x: x.strip('+').replace(',', ''))
    dataset['Price'] = dataset['Price'].apply(lambda x: x.strip('$'))
    dataset = dataset.drop(['Android Ver'], axis=1)
    dataset = dataset.drop(['Current Ver'], axis=1)
    dataset = dataset.drop(['Last Updated'], axis=1)
    dataset = dataset.drop(['Size'], axis=1)
    dataset = dataset.drop(['App'], axis=1)
    dataset = dataset.drop(['Type'], axis=1)

    targets=dataset.iloc[:,0].values
    dataset = dataset.drop(['Rating'], axis=1)

    #print(dataset.head())
    #print(dataset.info())

    sc = StandardScaler()
    toscale=dataset.iloc[:,0:3]
    #print(toscale)
    scaled=sc.fit_transform(dataset)
    dataset=dataset.drop(['Reviews'],axis=1)
    dataset=dataset.drop(['Installs'],axis=1)
    dataset=dataset.drop(['Price'],axis=1)
    #print(type(dataset))
    #print(type(scaled))
    scaled=pd.DataFrame({'Column1': scaled[:,0],'Column2': scaled[:,1],'Column3': scaled[:,2]})
    dataset=pd.concat([dataset,scaled],axis=1).drop_duplicates().reset_index(drop=True)
    #print(dataset.columns.values)
    print(len(targets))
    print(len(dataset))


    '''
    scaled_features = dataset.copy()
    col_names = ['Reviews', 'Installs','Price']

    features = scaled_features[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features[col_names] = features
    print(scaled_features.head())
    '''

    x_train, x_test, y_train, y_test = train_test_split(dataset, targets, test_size=0.2, random_state=0)





    model = build_model()

    #### fit the ANN to the training set

    batch_size = 10
    max_epochs = 50
    print(type(x_train)) # <class 'numpy.ndarray'>
    print(type(y_train))#<class 'numpy.ndarray'>

    model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs)

    #### predicting







if __name__ == '__main__':
    main()