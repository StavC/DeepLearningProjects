
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


def main():


    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    dataset = pd.read_csv('googleplaystore.csv')
    #print(dataset.head())
    dataset=dataset.dropna(axis='rows')
    inputs=dataset.iloc[:,1:10]
    inputs=inputs.drop(['Rating'],axis=1)
    inputs=inputs.drop(['Size'],axis=1)
    inputs=inputs.drop(['Type'],axis=1)
    inputs['Installs']=inputs['Installs'].apply(lambda x:x.strip('+').replace(',',''))
    #print(inputs.isnull().sum())## checking for nulls
    #print(inputs.head())
    targets=dataset.iloc[:,2:3] #targets are ready now
    #print(targets.head())
    print(targets.isnull().sum()) ### checking for nulls

    #### creating dummies for inputs
    categories=pd.get_dummies(inputs['Category'],drop_first=True)# categories dummies
    inputs=pd.concat([inputs,categories],axis=1)
    content_rating=pd.get_dummies(inputs['Content Rating'],drop_first=True)
    inputs=pd.concat([inputs,content_rating],axis=1)
    Genres = pd.get_dummies(inputs['Genres'], drop_first=True)
    inputs = pd.concat([inputs, Genres], axis=1)
    inputs=inputs.drop(['Category'],axis=1)
    inputs=inputs.drop(['Content Rating'],axis=1)
    inputs=inputs.drop(['Genres'],axis=1)

    print(inputs.head())











if __name__ == '__main__':
    main()