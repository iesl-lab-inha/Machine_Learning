## Pre-processing File
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def Preprocessing(classes,columns2,Wx,dx,data):
    drivers =[]
    for c in classes:
        drivers.append(data[data['Class']==c])
    dataa=[]
    for c in range(len(drivers)):
        nt=0
        nv=0
        drivers[c]=drivers[c].reset_index(drop=True)
        idxs=drivers[c][drivers[c]['Time(s)']==1].index.values
        for i in range(len(idxs)):
            if i <(len(idxs)-1):
                nt=nt+1
                dataa.append(drivers[c][idxs[i]:idxs[i+1]])
            if i==(len(idxs)-1):
                nv=nv+1
                dataa.append(drivers[c][idxs[i]:])
        #print("Driver : "+str(c)+" number of trips :"+str(len(idxs))+ "  For Train : "+str(nt)+"  For valid :"+str(nv))


    drivers=[]
    ss=0
    for i in range(len(dataa)):
        #print(n)
        n=int(len(dataa[i])/Wx)
        #print(n)
        #print(" Drive "+str(i)+" contains "+str(n)+" subdriversets")
        dd=0
        for j in range(n):
            #print(j)
            temp=dataa[i][dd:dd+Wx]
            #print(dd)
            temp=temp.reset_index(drop=True)
            drivers.append(temp)
            ss=ss+1
            dd=dd+dx
            #dd=dd+10
            #print("This is dd \t ",dd)
    print("total is "+str(ss))



    samples = list()
    labels=list()

    scaler = StandardScaler()
    scaler.fit(data[columns2].values)
    for c in drivers:
        labels.append(c['Class'][0])
        del c['Class']
        del c['Time(s)']
        samples.append(scaler.transform(c[columns2].values))
    data = np.array(samples)
    #print(data.shape)
    return data,labels

def Labels_Transform(labels):

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels=le.transform(labels)
    return labels

def Data_Split_Train_Val_Test(data,labels):

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=31)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=31)
    return X_train, X_test, y_train, y_test, X_val,y_val