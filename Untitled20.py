#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("C:/Users/Somay/Documents/heart.csv",encoding='ISO-8859-1')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


sns.countplot(df.target)


# In[8]:


plt.hist(df.age,bins=20)
plt.xlabel("age")
plt.ylabel("count")


# In[9]:


sns.pairplot(df)


# In[10]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),fmt="0%",annot=True)


# In[11]:


df_new=pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'],drop_first=True)


# In[12]:


df_new.columns


# In[13]:


df_new.head()


# In[14]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
columns_scaling=['age','trestbps','chol','thalach','oldpeak']
df_new[columns_scaling]=sc.fit_transform(df_new[columns_scaling])


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import train_test_split


# In[16]:


x=df_new.iloc[:,:-1]
print(x.shape)


# In[17]:


y=df_new.iloc[:,-1]
y.value_counts()


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",x_test.shape)


# In[19]:


rf=RandomForestClassifier(n_jobs=-1)
param_rf={'n_estimators':[50,100,200,250],'criterion':['gini','entropy'],'max_depth':[3,5,10,None],'min_samples_leaf':randint(1,3),'max_features':['auto','sqrt','log2'],'bootstrap':[True,False]}


# In[20]:


rf_random_cv=RandomizedSearchCV(rf,param_distributions=param_rf,cv=5,n_jobs=-1,n_iter=10)


# In[21]:


rf_random_cv.fit(x_train,y_train)


# In[22]:


print("the best score is",rf_random_cv.best_score_)
print("the best estimator is",rf_random_cv.best_estimator_)
print("the best params is",rf_random_cv.best_params_)


# In[23]:


d_tree=DecisionTreeClassifier()


# In[24]:


param_d_tree=param_rf={'criterion':['gini','entropy'],'max_depth':[3,5,10,None],'min_samples_leaf':randint(1,3),'max_features':['auto','sqrt','log2']}


# In[25]:


rscv_tree=RandomizedSearchCV(d_tree,param_distributions=param_d_tree,cv=5,n_jobs=-1,n_iter=10,return_train_score=False)


# In[26]:


rscv_tree.fit(x_train,y_train)


# In[27]:


print("the best score is",rscv_tree.best_score_)
print("the best estimator is",rscv_tree.best_estimator_)
print("the best params is",rscv_tree.best_params_)


# In[28]:


lr=LogisticRegression()


# In[29]:


lr_param={'penalty':['none'],'C':[1,0.1,0.0,1],'solver':['lbfgs']}


# In[30]:


rscv_lr=RandomizedSearchCV(lr,param_distributions=lr_param,n_jobs=-1,n_iter=10,cv=5,return_train_score=False)


# In[31]:


rscv_lr.fit(x_train,y_train)


# In[32]:


print('the best score',rscv_lr.best_score_)
print(' the best parameter is',rscv_lr.best_params_)


# In[33]:


print(pd.DataFrame([{'model':'RandomForest','Best Score':rf_random_cv.best_score_},{'model':'DecisionTree','Best Score':rf_random_cv.best_score_},{'model':'LogisticRegression','Best Score':rf_random_cv.best_score_}]))


# In[34]:


lr_new=LogisticRegression(penalty='none',C=1.0,solver='lbfgs')


# In[35]:


lr_new.fit(x_train,y_train)


# In[44]:


lr_new.score(x_test,y_test)


# In[48]:


y_pred=lr_new.predict(x_test)


# In[49]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print("classification report is")
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# In[51]:


plt.figure(figsize=(6,3))
conf= confusion_matrix(y_test,y_pred)
sns.heatmap(conf,annot=True,cmap='coolwarm')
plt.title('confusion matrix')


# In[ ]:




