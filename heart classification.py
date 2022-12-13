#!/usr/bin/env python
# coding: utf-8

# In[226]:


import pandas as pd


# In[227]:


import numpy as np#
import pandas as pd#
import matplotlib.pyplot as plt#
from matplotlib import rcParams#
from matplotlib.cm import rainbow#
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings#
warnings.filterwarnings('ignore')#


# In[228]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[148]:


data = pd.read_csv('heart.csv')


# In[149]:


data.info()


# In[150]:


data_dup = data.duplicated().any()


# In[151]:


data_dup


# In[152]:


data = data.drop_duplicates()


# In[153]:


data_dup


# In[154]:


cate_val = []
cont_val = []
for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)


# In[155]:


cate_val


# In[156]:


cont_val


# In[157]:


data['cp'].unique()


# In[158]:


cate_val.remove('sex')
cate_val.remove('target')
data = pd.get_dummies(data,columns = cate_val,drop_first=True)


# In[159]:


data.head()


# In[160]:


from sklearn.preprocessing import StandardScaler


# In[161]:


from sklearn.preprocessing import StandardScaler


# In[162]:


st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])


# In[163]:


data.head()


# In[164]:


ax = data.hist(figsize=(15,15))
plt.show()


# In[165]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[166]:


X = data.drop('target',axis=1)


# In[167]:


y = data['target']


# In[168]:


from sklearn.model_selection import train_test_split


# In[169]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=42)


# In[170]:


y_test


# In[171]:


data.head()


# In[172]:


from sklearn.linear_model import LogisticRegression


# In[173]:


log = LogisticRegression()
log.fit(X_train,y_train)


# In[174]:


y_pred1 = log.predict(X_test)


# In[175]:


from sklearn.metrics import accuracy_score


# In[176]:


accuracy_score(y_test,y_pred1)


# In[177]:


from sklearn import svm


# In[178]:


svm = svm.SVC()


# In[179]:


svm.fit(X_train,y_train)


# In[180]:


y_pred2 = svm.predict(X_test)


# In[181]:


accuracy_score(y_test,y_pred2)


# In[182]:


from sklearn.neighbors import KNeighborsClassifier


# In[183]:


knn = KNeighborsClassifier()


# In[184]:


knn.fit(X_train,y_train)


# In[185]:


y_pred3=knn.predict(X_test)


# In[186]:


accuracy_score(y_test,y_pred3)


# In[187]:


score = []

for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test,y_pred))


# In[188]:


score


# In[189]:


import matplotlib.pyplot as plt


# In[190]:


plt.plot(score)
plt.xlabel("K Value")
plt.ylabel("Acc")
plt.show()


# In[191]:


knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_test,y_pred)


# In[192]:


data = pd.read_csv('heart.csv')#non linear ml


# In[193]:


data = data.drop_duplicates()


# In[194]:


X = data.drop('target',axis=1)
y=data['target']


# In[195]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,
                                                random_state=42)


# In[196]:


from sklearn.tree import DecisionTreeClassifier


# In[197]:


dt = DecisionTreeClassifier()


# In[198]:


dt.fit(X_train,y_train)


# In[199]:


y_pred4= dt.predict(X_test)


# In[200]:


accuracy_score(y_test,y_pred4)


# In[201]:


from sklearn.ensemble import RandomForestClassifier


# In[202]:


rf = RandomForestClassifier()


# In[ ]:





# In[203]:


rf.fit(X_train,y_train)


# In[204]:


y_pred5= rf.predict(X_test)


# In[205]:


accuracy_score(y_test,y_pred5)


# In[206]:


from sklearn.ensemble import GradientBoostingClassifier


# In[207]:


gbc = GradientBoostingClassifier()


# In[208]:


gbc.fit(X_train,y_train)


# In[209]:


y_pred6 = gbc.predict(X_test)


# In[210]:


accuracy_score(y_test,y_pred6)


# In[211]:


final_data = pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],
                          'ACC':[accuracy_score(y_test,y_pred1)*100,
                                accuracy_score(y_test,y_pred2)*100,
                                accuracy_score(y_test,y_pred3)*100,
                                accuracy_score(y_test,y_pred4)*100,
                                accuracy_score(y_test,y_pred5)*100,
                                accuracy_score(y_test,y_pred6)*100]})


# In[212]:


final_data


# In[213]:


from sklearn.ensemble import RandomForestClassifier


# In[214]:


rf = RandomForestClassifier()
rf.fit(X,y)


# In[215]:


import pandas as pd


# In[216]:


new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
     'slope':2,
    'ca':2,
    'thal':3,    
},index=[0])


# In[217]:


new_data


# In[218]:


p = rf.predict(new_data)
if p[0]==0:
    print("No Disease")
else:
    print("Disease")


# In[219]:


import joblib


# In[220]:


joblib.dump(rf,'model_joblib_heart')


# In[221]:


model = joblib.load('model_joblib_heart')


# In[222]:


model.predict(new_data)


# In[223]:


data.tail()


# In[224]:


from tkinter import *
import joblib


# In[225]:


from tkinter import *
import joblib
import numpy as np
from sklearn import *
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p8,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="No Heart Disease").grid(row=31)
    else:
        Label(master, text="Possibility of Heart Disease").grid(row=31)
    
    
master = Tk()
master.title("Heart Disease Prediction System")


label = Label(master, text = "Heart Disease Prediction System"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)

Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter Value of CP").grid(row=3)
Label(master, text="Enter Value of trestbps").grid(row=4)
Label(master, text="Enter Value of chol").grid(row=5)
Label(master, text="Enter Value of fbs").grid(row=6)
Label(master, text="Enter Value of restecg").grid(row=7)
Label(master, text="Enter Value of thalach").grid(row=8)
Label(master, text="Enter Value of exang").grid(row=9)
Label(master, text="Enter Value of oldpeak").grid(row=10)
Label(master, text="Enter Value of slope").grid(row=11)
Label(master, text="Enter Value of ca").grid(row=12)
Label(master, text="Enter Value of thal").grid(row=13)



e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)



Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




