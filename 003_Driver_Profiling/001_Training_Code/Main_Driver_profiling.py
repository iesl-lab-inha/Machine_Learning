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
from Utilities import Read_data
from Pre_Processing import Preprocessing
from Pre_Processing import Labels_Transform
from Pre_Processing import Data_Split_Train_Val_Test
from Model_Creation import Model_setup
from Training_Model import Training_Model
from Testing_Model import Testing_with_Anomalies
from Utilities import Save_Pickle
from Utilities import Load_Pickle
from Utilities import Saving_Model
from datetime import date
import time



## to run this file , write in command line python Main_Driver_profiling.py 150 6 10
################################ where 150 here  is number of epochs, 6 is the degree shift, 10 is number classes

def main(argv):
    '''
        The Driver Profiling Deep learning architecture , following command-line arguments,

        @param epochs:  epochs are the number of iterations for training the model , more the epochs, will train to a better converged values, Recommended is epochs>=150
        @param dx:  It is the shift/jump between the moving window of features, the optimum values for the window size of 40, is 6, dx =6
        @param index_sp: It is the number


        This function read the CSV file containing features, process it with windowing and normalization, then apply deep-learning for the classification
        Utlimately, it provides accuracy, loss, and executation time of only evaluation funtion.
        It stores the model, history and results along with the time stamp

        '''


    '''
    Meanwhile only two arguments are used, later after making functions dynamic and model creation, we will use other agrguements too.
    '''

    epochs = int(argv[1]) #epochs = 150  ## number of epochs

    dx = int(argv[2])## dx=6, shift, opposite of overlap

    index_sp = int(argv[3])  # number of classes, must be 10, incase of 10, means 1:10, if 4 , 1:4


    #pdb.set_trace()


    '''
    ################################################################################## Reading Configuration file and parameters
    '''

    Config = Read_data('Config.csv')
    path = Config['Values'][5]+'.csv';
    # path = 'full_data_test.csv';

    Acol = Config['Values'][6]
    columns2 = Acol.split(',')
    #columns2=["Long_Term_Fuel_Trim_Bank1","Intake_air_pressure","Accelerator_Pedal_value","Fuel_consumption","Torque_of_friction","Maximum_indicated_engine_torque","Engine_torque","Calculated_LOAD_value",
    #"Activation_of_Air_compressor","Engine_coolant_temperature","Transmission_oil_temperature","Wheel_velocity_front_left-hand","Wheel_velocity_front_right-hand","Wheel_velocity_rear_left-hand",
    #"Torque_converter_speed"]

    Atemp = Config['Values'][1]
    classes_total =Atemp.split(",")
    # classes_total=['A','B','C','D','E','F','G','H','I','J']


    classes= classes_total[0:index_sp]
    #classes = Config['Values'][1]
    Ar = Config['Values'][2]
    rates = Ar.split(',')
    rates = [float(i) for i in rates]
    #rates = [0, 0.01, 0.1, 0.3, 0.5]  ## anomaly rate

    Aw = Config['Values'][3]
    rows = Aw.split(',')
    rows = [int(i) for i in rows]
    # rows = [1, 10]  ## anomaly window

    As = Config['Values'][4]
    sensors = As.split(',')
    sensors = [int(i) for i in sensors]
    #sensors = [7]  ## anomaly number of features or number of sensors in which anomaly you want to add


    #pdb.set_trace()
    '''
       ##################################################################### End of Configuration file 
    '''
    #### Making timestamps for saving the file with unique name
    # datetime object containing current date and time
    d1 = date.today()
    day = d1.day
    mon= d1.month
    yr = d1.year
    tt = int (time.time())
   # mcscnd = datetime.microsecond()
    Time_Stamp = str(yr) + str(mon) + str(day) + str(tt);

    print("now =", str(yr) + str(mon) + str(day) + str(tt) );


    Wx = 40 ## window size




    #print(epochs)




    data = Read_data(path)  ### reading the csv file

    data,labels = Preprocessing(classes,columns2,Wx,dx,data) ### normalzing, windowing,  etc

    labels = Labels_Transform(labels) ## label formatting , one hot encoding etc

    X_train, X_test, y_train, y_test, X_val,y_val =Data_Split_Train_Val_Test(data,labels)  ## splitting data into train, validation and test data

    model = Model_setup(data) ## creating a model based on data size (window size of each chunk)

    Trained_model,history = Training_Model(model,epochs,X_train,y_train,X_val,y_val)  ## training model till the epochs provided
    ## batch size is 32 , by default

    #accuracy,Loss,endtime = Testing_with_Anomalies(X_test,y_test,rows,sensors,rates,model)
    #Result_list = [accuracy, Loss, endtime]


    #print('{ "cut": %f, "fill": %f,"volume": %f }' % calculate_volume(tiff, geom, base_plane, custom_elevation))
    #print('{ "Accuracy": %f, "Loss": %f,"Execution_Time": %f }' % Testing_with_Anomalies(X_test,y_test,rows,sensors,rates,model) )
    Results = '{ "Accuracy": %f, "Loss": %f,"Execution_Time": %f }' % Testing_with_Anomalies(X_test,y_test,rows,sensors,rates,Trained_model);
    print(Results)
    Save_Pickle(Results, Time_Stamp + '_results')
    Save_Pickle(history.history, Time_Stamp + '_history')

    Save_Pickle(X_test,'X_test');
    Save_Pickle(y_test, 'y_test');

    Saving_Model(Trained_model, Time_Stamp + '_Proposed_LSTM_based')  ## variable , name and path(optional)
    #Saving_Model(model, Time_Stamp + '_Proposed_LSTM_based')  ## variable , name and path(optional)


if __name__ == '__main__':
    main(sys.argv)