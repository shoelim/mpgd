import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import airfoil_data 
import pandas as pd

# read and process data
def get_data(RANDOM_SEED=0):
    np.random.seed(RANDOM_SEED)
    df = pd.read_csv('airfoil_data/airfoil_self_noise.csv')
    #df.head()
    #df.tail()
    #df.info()
    #df.describe()
    X = df.drop('Scaled sound pressure level, in decibels', axis=1).values
    y = df['Scaled sound pressure level, in decibels'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    sc1 = StandardScaler()
    y_train = y_train.reshape(-1, 1)
    y_train = sc1.fit_transform(y_train)

    y_test = y_test.reshape(-1, 1)
    y_test = sc1.transform(y_test)

    X_train = np.array(X_train, dtype = np.float32)
    y_train = np.array(y_train, dtype = np.float32)
    X_test = np.array(X_test, dtype = np.float32)
    
    return X_train, y_train, X_test, y_test
