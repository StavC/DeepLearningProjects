import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import  LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler





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
    x_train,x_test,y_train,y_tes=train_test_split(inputs,targets,test_size=0.2,random_state=0)

    #### Feature Scaling
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)








if __name__ == '__main__':
    main()