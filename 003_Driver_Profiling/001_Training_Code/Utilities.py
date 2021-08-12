from tensorflow import keras
import pickle
import pandas as pd

def Saving_Model (model,name):
    model.save(name + '.h5')
    print("done- saving the model",name)


def Save_Pickle(history,name):
    pickle.dump(history, open(name +'.pkl','wb'))

def Load_Pickle(name):
    datapkle = open(name +'.pkl', 'rb')
    datapkle = pickle.load(datapkle)
    return datapkle

def Load_Model (name):
    #model.save(name + '.h5')
    new_model = keras.models.load_model(name + '.h5')
    #print("done- saving the model",name)
    return new_model

def Read_data(path):
    data=pd.read_csv(path)
    return data