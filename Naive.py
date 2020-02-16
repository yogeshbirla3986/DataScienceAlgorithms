#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np

files = glob.glob("/home/yogesh/Desktop/dataScience/review_polarity/txt_sentoken/*/*.txt")

vocab = {}
index = 0

for file in files:
    
    f = open(file,"r")
    lines = f.readlines()
    for line in lines:
        
        words = line.split()
        for word in words:
            
            if word in ['"','.','(',')',',']:
                pass
            elif word not in vocab:
                vocab[word] = index
                index += 1
    f.close()

print(index)


# 

# In[2]:


files = glob.glob("/home/yogesh/Desktop/dataScience/review_polarity/txt_sentoken/pos/*.txt")

v = index
pos_mat_train = np.zeros([700,v+1], float) + 10**(-5)

i = 0

for file in files[:700]:
    
    f = open(file,"r")
    lines = f.readlines()
    
    for line in lines:
        
        words = line.split()
        for word in words:
            
            if word in ['"','.','(',')',',']:
                pass
            else:
                pos_mat_train[i][vocab[word]] += 1
                
    i += 1
print(i)


# In[3]:


files = glob.glob("/home/yogesh/Desktop/dataScience/review_polarity/txt_sentoken/neg/*.txt")

neg_mat_train = np.zeros([700,v+1], float) + 10**(-5)

i = 0

for file in files[:700]:
    
    f = open(file,"r")
    lines = f.readlines()
    
    for line in lines:
        
        words = line.split()
        for word in words:
            
            if word in ['"','.','(',')',',']:
                pass
            else:
                neg_mat_train[i][vocab[word]] += 1
                
    i += 1

print(i)


# In[4]:


files = glob.glob("/home/yogesh/Desktop/dataScience/review_polarity/txt_sentoken/pos/*.txt")

mat_test = np.zeros([600,v+1], float) + 10**(-5)

i = 0

for file in files[700:]:
    
    f = open(file,"r")
    lines = f.readlines()
    
    for line in lines:
        
        words = line.split()
        for word in words:
            
            if word in ['"','.','(',')',',']:
                pass
            else:
                mat_test[i][vocab[word]] += 1
                
    i += 1
print(i)


# In[5]:


files = glob.glob("/home/yogesh/Desktop/dataScience/review_polarity/txt_sentoken/neg/*.txt")

for file in files[700:]:
    
    f = open(file,"r")
    lines = f.readlines()
    
    for line in lines:
        
        words = line.split()
        for word in words:
            
            if word in ['"','.','(',')',',']:
                pass
            else:
                mat_test[i][vocab[word]] += 1
                
    i += 1

print(i)


# In[6]:


pos_mat_count = np.zeros([v+1],float)

for i in range(v+1):
    for j in range(700):
        pos_mat_count[i] += pos_mat_train[j][i]

total_pos_count = sum(pos_mat_count)
print(total_pos_count)


# In[7]:


neg_mat_count = np.zeros([v+1],float)

for i in range(v+1):
    for j in range(700):
        neg_mat_count[i] += neg_mat_train[j][i]

total_neg_count = sum(neg_mat_count)
print(total_neg_count)


# In[9]:


import math

pos_mat_probablity = np.zeros([v+1] , float)

neg_mat_probablity = np.zeros([v+1] , float)

rem = 10**(-5)

for i in range(v+1):
   
    pos_mat_probablity[i] = math.log(pos_mat_count[i]+rem) - math.log(rem*(v+1)+total_pos_count)
    
    neg_mat_probablity[i] = math.log(neg_mat_count[i]+rem) - math.log(rem*(v+1)+total_neg_count)

print(pos_mat_probablity)
    


# In[11]:



# prediction of 600 files

classified = np.zeros([600],int)
result = np.zeros([600],int)

for i in range(600):
    
    pos_score = 0
    neg_score = 0
    
    for j in range(1+v):
        
        pos_score = pos_score + (mat_test[i][j])*pos_mat_probablity[j]
    
        neg_score = neg_score + (mat_test[i][j])*neg_mat_probablity[j]
    
    if(pos_score > neg_score):
        classified[i] = 1
    
    if(i < 300 and classified[i] == 1):
        result[i] = 1
    elif(i >=300 and classified[i] == 0):
        result[i] = 1
    else:
        result[i] = 0
        
correct_classified = sum(result)
acc = correct_classified/len(result)

print(acc)


# In[ ]:




