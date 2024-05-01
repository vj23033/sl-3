
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

df= pd.read_csv('Social_Network_Ads.csv')
df.head()	

df.info()
df.describe()

X=df.iloc[:[2,3]].values
Y=df.iloc[:,4].values

X
Y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
print("Training and testing split was successful.")


from sklearn.preprocessing  import 	StandardScalar
sc=StandardScalar()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_train


from sklearn.linear_model import LogisticRegression()
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_predict)
print(cm)
from sklearn.metrics import precision_recall_fscore_support
prf= precision_recall_fscore_support(y_test,y_predict)
print('precision:',prf[0])
print('Recall:',prf[1])
print('fscore:',prf[2])
print('support:',prf[3])

from sklearn.metrics import classification_report
cr= classification_report(y_test,y_predict)
print(cr)
