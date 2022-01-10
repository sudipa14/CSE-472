#!/usr/bin/env python
# coding: utf-8

# In[336]:


import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import Counter
import math
from sklearn.impute import SimpleImputer


# In[476]:


def activate(x,theta):
    z = np.dot(x,theta)
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def gradient_descent(x, y, theta, alpha, iterations,switch=0):
#     print(Counter(costt))
    i=0
#     print('hello')
#     y=y.reshape(len(x),1)
    
#     print(x.shape,y.shape,theta.shape)
    while i<iterations:
#         h=activate(x,theta)
        h=activate(x,theta)
        cc=((y-h)*(1-h**2))
#         print(np.argwhere(np.isnan(x)))
        theta=theta+(alpha/len(x))*np.dot(x.T,cc)
        y_pred=predict(x,theta)
        error=(np.sum(y-y_pred)**2).mean()
        if error<.5 and switch!=0:
            return theta
        i=i+1
    return theta 
def predict(X,theta) :    
    z=activate(X,theta)
    Y = np.where( z >= 0, 1.0, -1.0 )        
    return Y
def logistic_reg(x_train,y_train,theta,alpha,itr):
    theta = gradient_descent(x_train, y_train, theta, alpha,itr)
#     Y_pred =predict(x_test,theta)
    return theta
def entropy(df):
    probs = df[df.shape[1]-1].value_counts(normalize=True)
    
    entropy = -1 * np.sum(np.log2(probs) * probs)
    return entropy


# In[338]:


# Adaboost implement


# In[493]:


def adaboost(dataframe,k):
    list_w=[]
    list_z=[]
    y_target=dataframe[dataframe.shape[1]-1]
    x_train_first=dataframe.drop(dataframe.columns[dataframe.shape[1]-1], axis=1)
    x_train_or, x_test_or, y_train_or, y_test_or = train_test_split(x_train_first, y_target, test_size=0.2, random_state=0)
    df_n= pd.concat([x_train_or,y_train_or], axis=1)
    i=1
    y_train_or= y_train_or.to_numpy()
    y_train_or=np.reshape(y_train_or,(len(y_train_or),1))

    N=df_n.shape[0]
    W=[1/N]*N
    
    W=np.array(W)
    theta= np.zeros(((df_n.shape[1]-1),1))
    
    while i<=k:
        data=df_n.sample(n=df_n.shape[0],replace=True,weights=W)
        last=len(df_n.columns)-1
        y_target_sampled=data[last]
        y_target_sampled = y_target_sampled.to_numpy()
        y_target_sampled=np.reshape(y_target_sampled,(len(y_target_sampled),1))
        x_given_sampled=data.drop(data.columns[last], axis=1)
        weight_sample=logistic_reg(x_given_sampled,y_target_sampled,theta,.01,10000)
        list_w.append(weight_sample)
        Y_pred=predict(x_train_or,weight_sample)
        
        error=0
       
        for j in range(len(Y_pred)):
            if(Y_pred[j]!=y_train_or[j]):
                error+=W[j]
        print(error)
        if(error>.5):
            i=i+1
            continue
        error=min(error,1-.000000001)   
        for j in range(len(Y_pred)):
            if(Y_pred[j]==y_train_or[j]):
                W[j]=(W[j]*(error))/(1-error)
        W = W/np.sum(W) 
        
        z=np.log2((1-error)/error)
        list_z.append(z)
        Y_pred=z*Y_pred

        i=i+1
    return y_test_or,x_test_or,list_w,list_z
        


# In[361]:


def adaboost_train(dataframe,k):
    list_w=[]
    list_z=[]
    y_target=dataframe[dataframe.shape[1]-1]
    x_train_first=dataframe.drop(dataframe.columns[dataframe.shape[1]-1], axis=1)
    x_train_or, x_test_or, y_train_or, y_test_or = train_test_split(x_train_first, y_target, test_size=0.2, random_state=0)
    df_n= pd.concat([x_test_or,y_test_or], axis=1)
    i=1
    y_test_or= y_test_or.to_numpy()
    y_test_or=np.reshape(y_test_or,(len(y_test_or),1))

    N=df_n.shape[0]
    W=[1/N]*N
    
    W=np.array(W)
    theta= np.zeros(((df_n.shape[1]-1),1))
    
    while i<=k:
        data=df_n.sample(n=df_n.shape[0],replace=True,weights=W)
        last=len(df_n.columns)-1
        y_target_sampled=data[last]
        y_target_sampled = y_target_sampled.to_numpy()
        y_target_sampled=np.reshape(y_target_sampled,(len(y_target_sampled),1))
        x_given_sampled=data.drop(data.columns[last], axis=1)
        weight_sample=logistic_reg(x_given_sampled,y_target_sampled,theta,.01,1000)
        list_w.append(weight_sample)
        Y_pred=predict(x_test_or,weight_sample)
        
        error=0
       
        for j in range(len(Y_pred)):
            if(Y_pred[j]!=y_test_or[j]):
                error+=W[j]
        print(error)
        if(error>.5):
            i=i+1
            continue
        error=min(error,1-.000000001)   
        for j in range(len(Y_pred)):
            if(Y_pred[j]==y_test_or[j]):
                W[j]=(W[j]*(error))/(1-error)
        W = W/np.sum(W) 
        
        z=np.log2((1-error)/error)
        list_z.append(z)
        Y_pred=z*Y_pred

        i=i+1
    return y_train_or,x_train_or,list_w,list_z


# In[487]:



def accuracy(y_pred,y_test):
    count = 0
    correctly_classified=0
#     print(y_pred.shape,y_test.shape)
    y_test=list(y_test)
    while count<len(y_pred):
        if (y_test[count]==y_pred[count]):
            correctly_classified = correctly_classified + 1
        count=count+1
    print(correctly_classified/count)
def true_positive(y_pred,y_test):
    count=0
    positive=0
    y_test=list(y_test)
    while count<len(y_pred):
        if (y_test[count]==y_pred[count] and y_pred[count]==1):
            positive = positive + 1
        count=count+1
    print(positive/count)
    return (positive/count)
def true_negative(y_pred,y_test):
    count=0
    negative=0
    y_test=list(y_test)
    while count<len(y_pred):
        if (y_test[count]==y_pred[count] and y_pred[count]==-1):
            negative = negative + 1
        count=count+1
    print(negative/count)
def precision(y_pred,y_test):
    count=0
    t_pos=0
    f_pos=0
    y_test=list(y_test)
    while count<len(y_pred):
        if (y_test[count]==y_pred[count] and y_pred[count]==1):
            t_pos=t_pos+1
        elif (y_test[count]!=y_pred[count] and y_pred[count]==1):
            f_pos=f_pos+1
        count=count+1
    print(t_pos/(t_pos+f_pos))
    return t_pos/(t_pos+f_pos)
def false_dis_rate(y_pred,y_test):
#     result=precision(y_pred,y_test)
    print(1-(precision(y_pred,y_test)))
def f1_score(y_pred,y_test):
    a=(2*precision(y_pred,y_test)*true_positive(y_pred,y_test))/(precision(y_pred,y_test)+true_positive(y_pred,y_test))
    print(a)
    return a
def perf_measure(y_actual, y_hat):
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP+= 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==-1:
            TN += 1
        if y_hat[i]==-1 and y_actual[i]!=y_hat[i]:
            FN += 1
        # Sensitivity, hit rate, recall, or true positive rate
    TPR=TP/(TP+FN)
    print("true positive rate:"+str(TPR*100)+"%")
    # Specificity or true negative rate
    TNR=TN/(TN+FP) 
    print("true negative rate:"+str(TNR*100)+"%")
    # Precision or positive predictive value
    if(TP==FP==0):
        print("Precision:"+str(1*100)+"%")
    else:
        PPV = TP/(TP+FP)
        print("Precision:"+str(PPV*100)+"%")
    
    # False discovery rate
    if(TP==FP==0):
        FDR=0
    else:FDR=FP/(TP+FP)
    print("False discovery rate:"+str(FDR*100)+"%")
    # Overall accuracy
    ACC=(TP+TN)/(TP+FP+FN+TN)
    print("Accuracy:"+str(ACC*100)+"%")
    F1_score=2*TP/(2*TP+FP+FN)
    print("F1_score:"+str(F1_score*100)+"%")

    


# In[469]:


# Information gain
def info_gain(df_coppy,feature_num):
    gain_arr = {}
    total_entropy=entropy(df_coppy)
    
#     bins5 = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
#     col = pd.cut(df_coppy[5], bins5)
#     df_new_dummy.drop(df_new_dummy.columns[5], inplace=True, axis=1)
#     df_new_dummy.insert(loc=5, column=5, value=col_5)
    i=1
    while i<df_coppy.shape[1]-1:
        p=(df_coppy[i].unique())
        # bins5 = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

        # col_5 = pd.cut(df_new[5], bins5)

        # df_new_dummy.drop(df_new_dummy.columns[5], inplace=True, axis=1)

        # df_new_dummy.insert(loc=5, column=5, value=col_5)
        if(len(p)>10):
            j=0
            bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
            col_num= pd.cut(df_coppy[i], bins)
            df_coppy.drop(df_coppy.columns[i], inplace=True, axis=1)
            df_coppy.insert(loc=i, column=i, value=col_num)
            p=df_coppy[i].unique()
                
        j=0
        sum=0
        while j<len(p):
            unique_val_df=df_coppy[(df_coppy[i]) == p[j]]
            
            sum+=(entropy(unique_val_df))*unique_val_df.shape[0]/df_coppy.shape[0]
#             print(unique_val_df.shape[0],df_coppy.shape[0])
            j=j+1
        
        gain_arr[i]=total_entropy-(sum)
        
#         print(sum,i)
        i=i+1
    gain_arr=sorted(gain_arr.items(), key=lambda x: x[1], reverse=True)
# #     print(gain_arr)
    list_feature_index=[]
    list_feature_index.append(0)
    
    for index in range(feature_num):
        list_feature_index.append(gain_arr[index][0])
    
    return list_feature_index,gain_arr


# In[490]:



# dataset 1 reading 
df=pd.read_csv("customer_churn.csv")
columns=df.columns
df=df.dropna()
ffff= pd.concat([df['TotalCharges'].str.split()
                       .str[0]
                       .str.replace(',','').astype(float)], axis=1)
df.drop('TotalCharges', inplace=True, axis=1)
df.insert(loc=19, column='TotalCharges', value=ffff)
# df['TotalCharges']=ffff
df.replace(r'^\s*$', np.nan, regex=True)
# print(np.where(df.applymap(lambda x: x == ' ')))
mean_value=df['TotalCharges'].mean()
df['TotalCharges'].fillna(value=mean_value, inplace=True)
p=df.dtypes
object_list=[]
# print(p[19])
for i in range(len(p)):
    if p[i]=="object":
        object_list.append(columns[i])
        
# print(df.dtypes)
# print(object_list)
# print(df.head())
i=1
while i<len(object_list):
    
    df[object_list[i]] = df[object_list[i]].astype('category')
    temp =df[object_list[i]].cat.codes
    index_no = df.columns.get_loc(object_list[i])
    df.drop(object_list[i], inplace=True, axis=1)
    df.insert(loc=index_no, column=object_list[i], value=temp)
    i=i+1
    

min_max_scaler = preprocessing.MinMaxScaler()
df.drop('customerID', inplace=True, axis=1)
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
column_names=df.keys()
df = pd.DataFrame(x_scaled)


# dataset scaled & will apply logistic regression

y=df[19]
y=y.replace(0.0,-1.0)
x=df.drop(df.columns[19], axis=1)
# theta = [0]*len(x.columns)

ones = np.ones(x.shape[0])
df_new= pd.DataFrame()
df_new.insert(0,0,ones)
i=0
while i <(len(x.columns)):
    df_new.insert(i+1,i+1,x[i])
    i=i+1

x=df_new.copy(deep=True)

# /////////////////////information gain hisab
# gain_arr = {}
# total_entropy=entropy(df)
df_new.insert(len(x.columns),len(x.columns),y)
# print(total_entropy)
df_new_dummy=df_new.copy(deep=True)

list_feature_index,gain_arr=info_gain(df_new_dummy,6)


# In[492]:


###############                          Logistic regression  with dataset 1

x=x[list_feature_index]

theta= np.zeros((len(x.columns),1))
# print(theta.shape)
# theta= np.random.uniform(low=0.0, high=1.0, size=(len(x.columns),1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

y_test = y_test.to_numpy()
y_train = y_train.to_numpy()
y_train=np.reshape(y_train,(len(y_train),1))
y_test=np.reshape(y_test,(len(y_test),1))

theta = gradient_descent(x_test, y_test, theta, 0.01,10000)
# print(theta)
Y_pred =predict(x_train,theta)
perf_measure(y_train,Y_pred)


# In[494]:


###############                          Adaboostimplementation  with dataset 1

# df_new.insert(len(x.columns),len(x.columns),y)
final_list=list()

y_test_or,x_test_or,list_w,list_z=adaboost(df_new,5)
for i in range(len(list_z)):
    Y_pred_ada=predict(x_test_or,list_w[i])
    final_list.append(Y_pred_ada*list_z[i])
    
res=np.sum(final_list,axis=0)
res=np.where(res>=0, 1,-1)
accuracy(res,y_test_or)
# N=x.shape[0]
# # df_new.insert(len(x.columns),len(x.columns),y)
# final_list=list()

# y_train_or,x_train_or,list_w,list_z=adaboost_train(df_new,20)
# for i in range(len(list_z)):
#     Y_pred_ada=predict(x_train_or,list_w[i])
#     final_list.append(Y_pred_ada*list_z[i])
    
# res=np.sum(final_list,axis=0)
# res=np.where(res>=0, 1,-1)
# accuracy(res,y_train_or)

###############                          Adaboostimplementation  with dataset 1


# In[477]:


#preprocessing second data
def dataset2(df_2):
    dtypes_df2=df_2.dtypes
    df_2=df_2.replace(' ?', np.nan)
    # # print(df_2.isin([' ?']).any())
    nan_values = df_2.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df_2.columns[nan_columns].tolist()



    for i in range(len(columns_with_nan)):
    #     print(dtypes_df2[i])
        if(dtypes_df2[columns_with_nan[i]]=="object"):
            df_2[columns_with_nan[i]].fillna(df_2[columns_with_nan[i]].mode()[0], inplace=True)

        else:
            mean_value=df_2[columns_with_nan[i]].mean()
            df_2.fillna(value=mean_value, inplace=True)



    # print('hello')      
    ob_list=[1,3,5,6,7,8,9,13]
    for i in range(len(ob_list)):
        df_2[ob_list] = df_2[ob_list].astype('category')
        temp =df_2[ob_list[i]].cat.codes
        del df_2[ob_list[i]]
        df_2.insert(loc=ob_list[i], column=ob_list[i], value=temp)



    x_2 = df_2.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled_2= min_max_scaler.fit_transform(x_2)
    column_names=df_2.keys()
    df_2 = pd.DataFrame(x_scaled_2)




    # # scaling done
    y_2=df_2[14]
    y_2=y_2.replace(0.0,-1.0)


    x_2=df_2.drop(df_2.columns[14], axis=1)
    x_2_test=df_2.drop(df_2.columns[14], axis=1)
    ones = np.ones(x_2.shape[0])

    df_new_2= pd.DataFrame()
    df_new_2.insert(0,0,ones)
    i=0
    while i <(len(x_2.columns)):
        df_new_2.insert(i+1,i+1,x_2[i])
        i=i+1

    x_2=df_new_2.copy(deep=True)

    theta_2= np.zeros((len(x_2.columns),1))
    return theta_2,x_2,y_2


# In[478]:


# ###############################second data set
df_2=pd.read_csv("adult.data",header = None)
df_2_test=pd.read_csv("adult.test",header = None)

df_2=df_2.replace(' <=50K',-1)
df_2=df_2.replace(' >50K',1)
df_2_test=df_2_test.replace(' <=50K.',-1)
df_2_test=df_2_test.replace(' >50K.',1)

theta_2,x_2,y_2=dataset2(df_2)
theta_test,x_2_test,y_2_test=dataset2(df_2_test)

y_2 = y_2.to_numpy()
y_2_test= y_2_test.to_numpy()
y_2=np.reshape(y_2,(len(y_2),1))
y_2_test=np.reshape(y_2_test,(len(y_2_test),1))

theta_2= gradient_descent(x_2_test, y_2_test, theta_2, 0.01,10000)
# # print(theta)
Y_pred_2 =predict(x_2,theta_2)
perf_measure(y_2,Y_pred_2)


# In[482]:


# ############adaboost for dataset 2

final_list2=list()
df_new_2=x_2.copy(deep=True)
df_new_2.insert(len(x_2.columns),len(x_2.columns),y_2)

y_test_or2,x_test_or2,list_w2,list_z2=adaboost(df_new_2,5)

# y_train_or2,x_train_or2,list_w2,list_z2=adaboost_train(df_new_2,20)
############test set er accuracy ber korar jonyo
for i in range(len(list_z2)):
    Y_pred_ada2=predict(x_test_or2,list_w2[i])
    final_list2.append(Y_pred_ada2*list_z2[i])
res2=np.sum(final_list2,axis=0)
res2=np.where(res2>=0, 1,-1)
accuracy(res2,y_test_or2)
############test set er accuracy ber korar jonyo
############train set er accuracy ber korar jonyo
# for i in range(len(list_z2)):
#     Y_pred_ada2=predict(x_train_or2,list_w2[i])
#     final_list2.append(Y_pred_ada2*list_z2[i])
# res2=np.sum(final_list2,axis=0)
# res2=np.where(res2>=0, 1,-1)
# accuracy(res2,y_train_or2)
############train set er accuracy ber korar jonyo


# In[486]:


#third data set
df_3=pd.read_csv("creditcard.csv")

df_3.replace(r'^\s*$', np.nan, regex=True)

convert=df_3['Class']
convert=convert.replace(0,-1)
del df_3['Class']
df_3.insert(loc=30,column='Class',value=convert)
df_pos3=df_3[df_3['Class'] == 1]
df_neg3=df_3[df_3['Class'] == -1]
df_neg_sample3=df_neg3.sample(9000)


frames = [df_pos3,df_neg_sample3]

df3_small= pd.concat(frames)
df3_small=df3_small.sample(df3_small.shape[0])
x_3 = df3_small.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled_3= min_max_scaler.fit_transform(x_3)
column_names=df3_small.keys()
df3_small = pd.DataFrame(x_scaled_3)



print(df3_small.shape)
 # # scaling done
y_3=df3_small[30]
y_3=y_3.replace(0.0,-1.0)
x_3=df3_small.drop(df3_small.columns[30], axis=1)
ones3= np.ones(x_3.shape[0])

df_new_3= pd.DataFrame()
df_new_3.insert(0,0,ones3)
i=0
while i <(len(x_3.columns)):
    df_new_3.insert(i+1,i+1,x_3[i])
    i=i+1

x_3=df_new_3.copy(deep=True)

theta3= np.zeros((len(x_3.columns),1))
x_train3, x_test3, y_train3, y_test3 = train_test_split(x_3, y_3, test_size=0.2, random_state=0)
y_test3 = y_test3.to_numpy()
y_train3 = y_train3.to_numpy()
y_train3=np.reshape(y_train3,(len(y_train3),1))
y_test3=np.reshape(y_test3,(len(y_test3),1))

theta3 = gradient_descent(x_train3, y_train3, theta3, 0.001,10000)
# # print(theta)
Y_pred3 =predict(x_test3,theta3)
perf_measure(y_test3,Y_pred3)



# In[484]:


#########################adaboost for dataset3

N=x_3.shape[0]
# df_new.insert(len(x.columns),len(x.columns),y)
final_list3=list()
df_new_3=x_3.copy(deep=True)
df_new_3.insert(len(x_3.columns),len(x_3.columns),y_3)

# y_test_or3,x_test_or3,list_w3,list_z3=adaboost(df_new_3,5)
y_train_or3,x_train_or3,list_w3,list_z3=adaboost_train(df_new_3,5)
# for i in range(len(list_z3)):
#     Y_pred_ada3=predict(x_test_or3,list_w3[i])
#     final_list3.append(Y_pred_ada3*list_z3[i])
# res3=np.sum(final_list3,axis=0)
# res3=np.where(res3>=0, 1,-1)
# accuracy(res3,y_test_or3)
for i in range(len(list_z3)):
    Y_pred_ada3=predict(x_train_or3,list_w3[i])
    final_list3.append(Y_pred_ada3*list_z3[i])
res3=np.sum(final_list3,axis=0)
res3=np.where(res3>=0, 1,-1)
accuracy(res3,y_train_or3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




