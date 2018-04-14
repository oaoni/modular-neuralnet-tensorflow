from ModularNeuralNet import ModularNeuralNet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def upload_data():
    #Load Data
    #Gene expression data
    data = pd.read_csv("./data/gene_expression_clinical.csv",index_col=0)

    #Remove non-relavant columns
    data.drop(data.columns[[60483,60484,60485,60486,60488,60489,60490,60491,60492,60493,60494,60495]],axis=1,inplace=True)
    data.head()

    y = np.zeros((data.shape[0],2))
    y[:,0] = data['sample_type'].values
    y[:,1] = [1 if x==0 else 0 for x in y[:,0]]
    Y = y
    #Zero in the first field is benign, and zero in the second field is malignant


    pca = PCA(n_components=20) #Make a PCA object with n = 15 PCs
    pca.fit(data.drop(['sample_type'],axis=1))

    pc = pca.transform(data.drop(['sample_type'],axis=1))

    #Shuffle the data set
    X,Y = shuffle(pc,Y,random_state=1)

    #Convert the dataset into train and test set
    train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.30, random_state=415)

    return train_x, test_x, train_y, test_y

if __name__ == '__main__':

    train_x, test_x, train_y, test_y = upload_data()

    #Initiate class
    #model = ModularNeuralNet(hidden_nodes=[25,25,25,25],activations=['tanh','relu','relu','sigmoid'],
                             #lr=0.1,graph=False,save_model=False,plots=True,epochs=10000,print_iters=False,
                             #save_step=5,print_final=True,input_dim=20)

    #Fit to training data
    #model.fit(train_x, train_y)

    #Classification agreement of test set
    #model.predict(train_x)

    #print(model.score(train_x,train_y))
