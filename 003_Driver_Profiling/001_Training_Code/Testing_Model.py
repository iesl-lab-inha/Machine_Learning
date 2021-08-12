import pandas as pd
import random
import time
def Testing_with_Anomalies(X_test,y_test,rows,sensors,rates,model):
    results=pd.DataFrame(columns=["Rate","Rows","Sensors","Accuracy"])
    for rate in rates:
          for row in rows:
            for sensor in sensors:
                X_test2=X_test.copy()
                total=X_test.shape[0]*X_test.shape[1]*X_test.shape[2]
                total=int(311*rate)
                n=sensor
                nrows=row
                for i in range(total):
                    if(i%10000==0):
                        print(i)
                    fst=random.randint(0,X_test.shape[0]-1)
                    snd=random.randint(0,X_test.shape[1]-nrows)
                    trh2=random.sample(range(0, X_test.shape[2]), n)
                    for j in range(n):
                        for jj in range(nrows):
                            X_test2[fst][snd+jj][trh2[j]]=X_test2[fst][snd+jj][trh2[j]]+3000
                acc=round(model.evaluate(X_test2, y_test)[1]*100,2)
                results = results.append({'Rate': str(rate*100)+"%",'Rows': str(row)+"",'Sensors': str(sensor)+"",'Accuracy': acc}, ignore_index=True)
    #model.evaluate(X_test2, y_test)
    st =time.time()
    Loss_Acc = model.evaluate(X_test, y_test);
    endtime = time.time()-st
    accuracy = round(Loss_Acc[1] * 100, 2)
    Loss =Loss_Acc[0];

    print(results)
    return accuracy,Loss,endtime
