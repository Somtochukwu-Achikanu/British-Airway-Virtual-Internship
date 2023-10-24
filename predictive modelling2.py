import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,recall_score,precision_score,confusion_matrix,classification_report
from sklearn.model_selection import RandomizedSearchCV,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
from IPython.display import display 
from random import randint

bookings = pd.read_csv('/Users/somtoachi-kanu/Downloads/customer_booking.csv', encoding ="ISO-8859-1" )
print(bookings.info())

flight_day = bookings['flight_day'].unique()
print(flight_day)

#To map nnumbers to the unique days
mapping = {'Mon': 1,
           'Tue': 2,
           'Wed': 3,
           'Thu': 4,
           'Fri': 5,
           'Sat': 6,
           'Sun': 7,
           }

#To change the flight day column
bookings['flight_day']=bookings['flight_day'].map(mapping)
print(bookings['flight_day'].unique())
     


#Training and testing models
X = bookings.drop('booking_complete', axis = 1)
y = bookings['booking_complete']

#changing object dtype to int dtype
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()
print(X.dtypes)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=21,stratify=y)

#Instantiate the model
rcf = RandomForestClassifier()

#fit the trained data into the classifer
rcf.fit(X_train,y_train)

#prediction
y_pred = rcf.predict(X_test)

#accuracy
accuracy =  accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)

#Hyperparameter tuning
param_distribution ={'n_estimators': range(50,500,10),
                     'max_depth': range(1,20,2)}

#create a classifier
rcf = RandomForestClassifier()

param = RandomizedSearchCV(rcf, param_distribution, cv =5)

#fit the trained set into param
param.fit(X_train,y_train)

#create variable for model
best_estimator = param.best_estimator_

print("the best estimator:", best_estimator)
#Prediction
y_pred = best_estimator.predict(X_test)

#confusion Matrix
cm = confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(cm).plot()

#Compute resulting matrix
cr = classification_report(y_test,y_pred)
print(cr)

#Further tuning
rcf = RandomForestClassifier()
rcf.fit(X_train,y_train)
y_pred = rcf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
recallscore = recall_score(y_test,y_pred)
precisionscore = precision_score(y_test,y_pred)
print('accuracy score:', accuracy,
      'recall score:', recallscore,
      'prediction score:', precisionscore)