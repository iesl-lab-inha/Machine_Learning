'''
Main entry point
'''
import sys

import tensorflow as tf
import pandas as pd
#matplotlib inline
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import random

from Testing_Model import Testing_with_Anomalies
from Utilities import Save_Pickle
from Utilities import Load_Pickle
from Utilities import Load_Model
from datetime import date
import time



## to run this file , write in command line python Main_Driver_profiling.py Proposed_LSTM_based X_test y_test
################################

def main(argv):
    '''
        The Driver Profiling Deep learning architecture , following command-line arguments,

        @param Name_of_Model:  Trained model name along with path, ".h5" format ,
        @param X_test:         Test data link or name of file, cotaining data in ".pkl" format
        @param y_test:         Corresponding test lables, name with path ,


        This function read the CSV file containing features, process it with windowing and normalization, then apply deep-learning for the classification
        Utlimately, it provides accuracy, loss, and executation time of only evaluation funtion.
        It stores the model, history and results along with the time stamp

        '''


    '''
    Meanwhile only two arguments are used, later after making functions dynamic and model creation, we will use other agrguements too.
    '''

    Name_of_Model = (argv[1]) # Name of model
    X_test = (argv[2]) ## name of test file including "data"
    y_test = (argv[3]) ## name of test file with "labales"
    '''
    ## Global variables
    '''
    #if len(argv)>1:
   # path = 'full_data_test.csv';
    #columns2=["Long_Term_Fuel_Trim_Bank1","Intake_air_pressure","Accelerator_Pedal_value","Fuel_consumption","Torque_of_friction","Maximum_indicated_engine_torque","Engine_torque","Calculated_LOAD_value",
    #"Activation_of_Air_compressor","Engine_coolant_temperature","Transmission_oil_temperature","Wheel_velocity_front_left-hand","Wheel_velocity_front_right-hand","Wheel_velocity_rear_left-hand",
    #"Torque_converter_speed"]
    #classes=['A','B','C','D','E','F','G','H','I','J']
    rates = [0, 0.01, 0.1, 0.3, 0.5]  ## anomaly rate
    rows = [1, 10]  ## anomaly window
    sensors = [7]  ## anomaly number of features or number of sensors in which anomaly you want to add
    '''
       ## End of Global variables, later, we will transform it to config files or pass it as arguements
    '''

    #### Making timestamps for saving the file with unique name
    # datetime object containing current date and time
    d1 = date.today()
    day = d1.day
    mon= d1.month
    yr = d1.year
    tt = int (time.time())

    Time_Stamp = str(yr) + str(mon) + str(day) + str(tt);

    print("now =", str(yr) + str(mon) + str(day) + str(tt) );


    #Wx = 40 ## window size




    #print(epochs)



    Trained_model = Load_Model(Name_of_Model)
    X_test = Load_Pickle(X_test)  ### reading the X_test data
    y_test = Load_Pickle(y_test)  ## reading X-Test corresponding  labels

    #data,labels = Preprocessing(classes,columns2,Wx,dx,data) ### normalzing, windowing,  etc

    #labels = Labels_Transform(labels) ## label formatting , one hot encoding etc

    #X_train, X_test, y_train, y_test, X_val,y_val =Data_Split_Train_Val_Test(data,labels)  ## splitting data into train, validation and test data

    #model = Model_setup(data) ## creating a model based on data size (window size of each chunk)

    #Trained_model,history = Training_Model(model,epochs,X_train,y_train,X_val,y_val)  ## training model till the epochs provided
    ## batch size is 32 , by default

    #accuracy,Loss,endtime = Testing_with_Anomalies(X_test,y_test,rows,sensors,rates,model)
    #Result_list = [accuracy, Loss, endtime]


    #print('{ "cut": %f, "fill": %f,"volume": %f }' % calculate_volume(tiff, geom, base_plane, custom_elevation))
    #print('{ "Accuracy": %f, "Loss": %f,"Execution_Time": %f }' % Testing_with_Anomalies(X_test,y_test,rows,sensors,rates,model) )
    Results = '{ "Accuracy": %f, "Loss": %f,"Execution_Time": %f }' % Testing_with_Anomalies(X_test,y_test,rows,sensors,rates,Trained_model);
    print(Results)
    Save_Pickle(Results, Time_Stamp + '_results')
    #Save_Pickle(history.history, Time_Stamp + '_history')

    #Save_Pickle(X_test,'X_test');
    #Save_Pickle(y_test, 'y_test');

    #Saving_Model(Trained_model, Time_Stamp + '_Proposed_LSTM_based')  ## variable , name and path(optional)
    #Saving_Model(model, Time_Stamp + '_Proposed_LSTM_based')  ## variable , name and path(optional)


if __name__ == '__main__':
    main(sys.argv)