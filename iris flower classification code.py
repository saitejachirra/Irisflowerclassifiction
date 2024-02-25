#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


import numpy as np 
import scipy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# # analyzing data with python

# In[3]:


dataset_read=pd.read_csv("Iris.csv",sep=",")
dataset_read


# In[4]:


dataset_read.head(10)


# In[5]:


dataset_read.tail(10)


# # Understand the dataset deeply

# In[6]:


dataset_read.describe()


# In[7]:


dataset_read.columns


# In[8]:


dataset_read.dtypes


# In[9]:


dataset_read.nunique


# In[10]:


dataset_updated=dataset_read.rename(columns={'SepalLengthCm':'sepallengthcm','SepalWidthCm':'sepalwidthcm','PetalLengthCm':'petallengthcm','PetalWidthCm':'petalwidthcm','Species':'class'})
print(dataset_updated)


# In[11]:


#plotting our first graph by using seaborn library


# In[12]:


sns.lineplot(x=dataset_updated["sepallengthcm"],y=dataset_updated["petallengthcm"])


# # Understanding the relationship between sepal length and petal length through scatterplot

# In[13]:


sns.scatterplot(x=dataset_updated["sepallengthcm"],y=dataset_updated["petallengthcm"])
plt.title("sepal length vs petal length")


# # Realtionship between sepal width and petal width with maplotlib

# In[14]:


plt.plot(dataset_updated['sepalwidthcm'])
plt.plot(dataset_updated['petalwidthcm'])
plt.legend(["sepalwidthcm","petalwidthcm"])


# In[15]:


plt.plot(dataset_updated['sepalwidthcm'])
plt.plot(dataset_updated['petalwidthcm'])
plt.plot(dataset_updated['sepallengthcm'])
plt.plot(dataset_updated['petallengthcm'])
plt.legend(["sepalwidthcm","petalwidthcm","sepallengthcm","petalwidthcm"])
plt.rcParams["figure.figsize"]=(18,5)


# In[16]:


sns.lmplot(x="sepallengthcm",y="sepalwidthcm",data=dataset_updated,hue='class',legend=False)
plt.legend("sepalwidthcm","sepallengthcm")


# In[17]:


fig,axes=plt.subplots(1,4,figsize=(20,5))
dataset_updated['sepallengthcm'].hist(ax=axes[0],color="r").set_title("sepallengthcm")
dataset_updated['petallengthcm'].hist(ax=axes[1],color="b").set_title("petallengthcm")
dataset_updated['sepalwidthcm'].hist(ax=axes[2],color="g").set_title("sepallengthcm")
dataset_updated['petalwidthcm'].hist(ax=axes[3],color="m").set_title("petallengthcm")


# In[18]:


fig,axes=plt.subplots(2,2,figsize=(16,5))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=1)
sns.barplot(x=dataset_updated["class"],y=dataset_updated["sepallengthcm"],palette='cool',ax=axes[0][0]).set_title('class vs sepallengthcm')
sns.barplot(x=dataset_updated["class"],y=dataset_updated["sepalwidthcm"],palette='cool',ax=axes[1][0]).set_title('class vs sepalwidthcm')
sns.barplot(x=dataset_updated["class"],y=dataset_updated["petallengthcm"],palette='CMRmap_r',ax=axes[0][1]).set_title('class vs petallengthcm')
sns.barplot(x=dataset_updated["class"],y=dataset_updated["petalwidthcm"],palette='CMRmap_r',ax=axes[1][1]).set_title('class vs petalwidthcm')


# # Distribution of species in thr dataset in %

# In[19]:


dataset_updated['class'].value_counts().plot.pie(explode=[0.04,0.04,0.04],shadow=True,autopct='%1.2f%%',colors=["lightcoral","lightpink","lightblue"]).set_title("iris species classifications")


# In[20]:


dataset_read.tail(10)


# # MODEL IMPLEMENTATION STARTS

# In[21]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
dataset_updated['class']=label_encoder.fit_transform(dataset_updated['class'])
dataset_updated['class'].unique()


# In[22]:


dataset_updated["class"].head()


# # SPLITTING THE DATA

# In[23]:


from sklearn.model_selection import train_test_split
x=dataset_updated.drop(['class'],axis=1)
y=dataset_updated['class']


# In[24]:


x_sepal_train,x_sepal_test,y_class_train,y_class_test=train_test_split(x,y,random_state=0,test_size=0.3)
x_sepal_train.shape,x_sepal_test.shape,y_class_train.shape,y_class_test.shape


# In[25]:


x_sepal_train


# # SUPERVISED MACHINE LEARNING ALGORITHMS

# In[26]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_sepal_train,y_class_train)
y_class_predicted=model.predict(x_sepal_test)


# # ACCURACY OF LINEAR REGRESSION MODEL

# In[27]:


from sklearn.metrics import accuracy_score
sc_lr=round(model.score(x_sepal_test,y_class_test)*100,2)
print("Accuracy:",str(sc_lr),"%")


# In[28]:


from sklearn import linear_model
logistic_model=linear_model.LogisticRegression(max_iter=130)
logistic_model.fit(x_sepal_train,y_class_train)


# In[29]:


y_class_logistic_predicted=logistic_model.predict(x_sepal_test)


# In[30]:


sc_logr=round(logistic_model.score(x_sepal_test,y_class_test)*100,2)
print("Accuracy:",str(sc_logr),"%")


# In[33]:


from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_class_test,y_class_test)
cm


# In[34]:


sns.heatmap(cm,annot=True,cmap='BuPu_r')
plt.rcParams["figure.figsize"]=(10,2)
plt.xlabel('predicted value')
plt.ylabel('class')


# In[38]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
#Instantiate learning model(k=3)
knn_model = KNeighborsClassifier(n_neighbors=3)
#Fitting the model
knn_model.fit(x_sepal_train, y_class_train)
#Predicting the Test set results
y_knn_pred=knn_model.predict(x_sepal_test)
accuracy=accuracy_score(y_class_test, y_knn_pred)*100
print('Accuracy of our model is equal'+str(round(accuracy,2))+'%.')
from sklearn.metrics import classification_report
print(classification_report(y_class_test,y_knn_pred))


# In[39]:


knn_cm=confusion_matrix(y_class_test, y_knn_pred)
knn_cm


# In[40]:


sns.heatmap(knn_cm,annot=True,cmap='BuPu_r')
plt.rcParams["figure.figsize"]=(10,2)
plt.xlabel('predicted value')
plt.ylabel('class')


# In[49]:


scores_plt=[sc_lr,sc_logr,accuracy]
algorithms=["Linear Regression","Logistic Regression","KNN"]
sns.set(rc={'figure.figsize':(11,6)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(x=algorithms,y=scores_plt)

