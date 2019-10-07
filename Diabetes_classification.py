import pandas as pd

url='C:/Users/Asus/Desktop/Tp1 ML/diabetes.csv'
dataset= pd.read_csv(url)
#print(dataset.head(20))
label= dataset['Outcome']
data=dataset.drop(['Outcome'],axis=1)   #axis=1 : colonne / axis=0 : ligne
x=dataset['Outcome'].value_counts()  #to know occurance in each class => see which model to choose
#print(x)

from sklearn.model_selection import train_test_split

x_train1,x_test1,train_label,test_label=train_test_split(data,label,test_size=0.33,random_state=0)

from sklearn import svm

clf=svm.SVC(kernel='linear', C=1) #C is a cte for the linear kernel the smaller the better

import time
start_time = time.time()


clf.fit(x_train1,train_label)  #fit will do teh train on the train dataset
train_time=time.time() - start_time

y_pred_test=clf.predict(x_test1)   #we test with the trained model on a data without labes to get a label

#to evaluate performance 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

acc=accuracy_score(test_label, y_pred_test)

CM=confusion_matrix(test_label,y_pred_test)
print(CM)

fpr, tpr, thresholds = roc_curve (test_label, y_pred_test,pos_label=1)

roc_auc = auc(fpr, tpr)

""" plot Roc curve"""
import matplotlib.pyplot as plt 
plt.plot(fpr, tpr, color='blue')
plt.title('Receiver operation Characteritic')
plt.plot(fpr, tpr, 'b' , label= 'AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')  #courbe en rouge du %50 
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylable('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#metrics per class for model performance 
from sklearn.metrics import classification_report
print(classification_report(test_label, y_pred_test))

#conclusion : the unbalanced class leads us to use the classifier  
#one class instead of binair 
#we can change also parameter or kerels of the model like sigmoid, ploy...
