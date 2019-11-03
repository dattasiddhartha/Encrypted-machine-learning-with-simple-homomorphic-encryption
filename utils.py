import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn import linear_model

from hme import encryption_train, decryption_train, encryption_test, decryption_test, estimator_OLS, predict

def dataload(cci_data, input_cols, output_cols):

    x_data = cci_data[input_cols]#.ix[0:5000,:]
    y_data = cci_data[output_cols]#.ix[0:5000]

    x_data_shortened = cci_data[input_cols].ix[0:50,:]
    y_data_shortened = cci_data[output_cols].ix[0:50]

    # H_enc = encryption_train(x_data,y_data) # encrypting the data
    H_enc = encryption_train(x_data_shortened,y_data_shortened) # encrypting the data

    H_dec = decryption_train(H_enc[0],H_enc[1],H_enc[2],H_enc[3])

    X_enc = H_enc[0]
    y_enc = H_enc[1]

    # https://www.cs.cmu.edu/~rjhall/JOS_revised_May_31a.pdf
    # source for use of orthogonal matrices and invertible matrices transformation for homomorophic encryption
    
    return x_data, y_data, x_data_shortened, y_data_shortened, X_enc, y_enc, H_enc