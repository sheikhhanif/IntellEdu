#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
student_names = ["Amirul", "Asraf", "Izzati", "Noor", "Safiyah", "Leo", 
                 "Yang", "Hao", "Fuu", "Ryan", "Anuradha", "Lutfi", "Lazim",
                 "Aminah", "Shahrukh", "Subramaniam", "Ahmad", "Hanif", "Mior",
                 "Nadh", "Amalina", "Muaath", "Dan", "Omar", "Fawzy", "Tinir", "Aishah",
                 "Jini", "Khaled", "Ilham", "Asma", "Muna", "Hazwani", "Lily", "Liyana",
                 "Hafizah", "Suraidah", "Reza", "Monir", "Akila"]

def predict_academic_path():
    data = pd.read_csv('Edu.csv')
    indvar = data.drop('School', axis= 1).values
    depvar = data.School.values
    x_train, x_test, y_train, y_test = train_test_split(indvar, depvar, test_size=0.25, random_state = 0)
    clf = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0)
    modela = clf.fit(x_train,y_train)
    y_pred = modela.predict(x_test)
    stu_path = y_pred[:40]
    academic_student_path = pd.DataFrame({'names': student_names,
                               'Academic Path': stu_path,
                              })
    academic_student_path.to_csv('future_student_path.csv')
    

predict_academic_path()




