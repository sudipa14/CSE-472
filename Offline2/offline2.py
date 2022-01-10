import pandas as pd
import numpy as np
import math
import collections
import random
from numpy import unravel_index
random.seed(3000)



data = pd.read_csv('data.txt',header=None)
viterbi_val_without=pd.read_csv('states_Viterbi_wo_learning.txt',header=None)
viterbi_val_after=pd.read_csv('states_Viterbi_after_learning.txt',header=None)
viterbi_val_after=viterbi_val_after[0]
viterbi_val_without=viterbi_val_without[0]
y=data[0]

def string_to_float(str):
    str_to_float=[]
    for element in str:
        str_to_float.append(float(element))
    return str_to_float
def pdf(miu,sigma,x):
    result=(1/(sigma*math.sqrt(2*math.pi)))
    expo=(-.5)*((float(x)-float(miu))/float(sigma))**2
    expo=math.e**expo
    return result*expo
def forward_algo(tran,emission,y,start_prob,n):
    forward=np.ones((n,len(y)))
    sum_zero=0
    for i in range(n):
        forward[i,0]=(start_prob[i]*pdf(mean[i],stand_dev[i],y[0]))
        sum_zero+=forward[i,0]
    for i in range(n):
        forward[i,0]=forward[i,0]/sum_zero

    for j in range(1,len(y)):
        sumf=0
        for i in range(n):      
            first=forward[:,j-1]*(transition[:,i])*(emission[i,j])
            forward[i,j]=sum(first)
            sumf+=forward[i,j]
        for i in range(n):
            forward[i,j]=forward[i,j]/sumf

    return forward
def backward_algo(tran,emission,y,start_prob,n):
    backward=np.ones((n,len(y)))
    for i in range(n):
        backward[i,len(y)-1]=1
    j=len(y)-2
    while j>=0:
        sumb=0
        for i in range(n):      
            first=backward[:,j+1]*(transition[i,:])*(emission[:,j])
            backward[i,j]=sum(first)
            sumb+=backward[i,j]
        
        for i in range(n):
            backward[i,j]=backward[i,j]/sumb
        j=j-1
    
    return backward
def calculate_emission(mean,stand_dev,y):
    B=[]
    for i in range(n):
        temp=[]
        for elem in y:
            res=pdf(mean[i],stand_dev[i],elem)
            temp.append(res)
        B.append(temp)
    B=np.array(B)
    return B
def transition_converge(tran,prev_tran):
    # print(tran,prev_tran)
    for i in range(n):
        for j in range(n):
            if(round(tran[i,j],7)!=round(prev_tran[i,j],7)):
                return -1
    return 1
def calculate_pi_star(n,y,f,b,f_sink):
    pi_star=np.ones((n,len(y)))
    for j in range(len(y)):
        sum_pi=0
        for ii in range(n):
            val=f[ii,j]*b[ii,j]
            pi_star[ii,j]=val/f_sink
            sum_pi+=pi_star[ii][j]
        for ii in range(n):
            pi_star[ii,j]=pi_star[ii,j]/sum_pi
    return pi_star
def calculate_pi_two_star(f,b,transition,emission,n,y):
    pi_starrr=np.ones((n,n,len(y)-1))
    for j in range(len(y)-1):
        sum_starr=0
        for k in range(n):
            for l in range(n):
                pi_starrr[k,l,j]=f[k,j]*transition[k,l]*emission[l,j+1]*b[l,j+1]
                sum_starr+=pi_starrr[k,l,j]
        for k in range(n):
            for l in range(n):
                pi_starrr[k,l,j]=pi_starrr[k,l,j]/sum_starr
    return pi_starrr

def baum_welch(tran,y,start_prob,n,mean,stand_dev,itr=1000):
    prev_tran=tran.copy()
    prev_mean=mean.copy()
    prev_stand_dev=stand_dev.copy()
    iteration=0
    while True:
        print(iteration+1,"iteration done")
       
        if iteration!=0:
            # check if transition matrix has already converged
            if(transition_converge(tran,prev_tran)==1):
                break
            else:
                prev_tran=tran.copy()
        iteration=iteration+1
        emission=calculate_emission(mean,stand_dev,y)
        start_prob=calculate_stationary(transition,n)
        # calculate new emission
        f=forward_algo(tran,emission,y,start_prob,n)
        # print(f)
        b=backward_algo(tran,emission,y,start_prob,n)
        # print(b)
        f_sink=0
        for ii in range(n):
            f_sink+=f[ii,len(y)-1]
        
        #pi_star
        pi_one_star=calculate_pi_star(n,y,f,b,f_sink)
        
        #pi_star_star
        pi_starrr=calculate_pi_two_star(f,b,transition,emission,n,y)
        # f,b,transition,emission,n,y
        # pi_double_star

        # transition matrix update korbo
        for k in range(n):
            sum_row=0
            for l in range(n):
                tran[k,l]=sum(pi_starrr[k,l,:])
                sum_row+=tran[k,l]
            for l in range(n):
                tran[k,l]=tran[k,l]/sum_row
        
        
        # miu ber korbo
        sum_y=0
        sum_k=0
        for k in range(n):
            for j in range(len(y)):
                sum_y+=pi_one_star[k,j]*y[j]
                sum_k+=pi_one_star[k,j]
            mean[k]=sum_y/sum_k
            
            sum_y=0
            sum_k=0
        #stand_dev ber korbo

        sum_y=0
        sum_k=0
        for k in range(n):
            for j in range(len(y)):
                sum_y+=pi_one_star[k,j]*(y[j]-mean[k])**2
                sum_k+=pi_one_star[k,j]
            stand_dev[k]=math.sqrt(sum_y/sum_k)
            sum_y=0
            sum_k=0
        
        # print(tran,mean,stand_dev)
    return tran,mean,stand_dev
        
        
def viterbi(n,y,start_prob,transition,mean,stand_dev,exact_val,s):
    # initialise matrices
    X=np.ones((n,len(y)))
    # X will contain maximum probablity
    X_pointer=np.ones((n,len(y)))
    for i in range(n):
        X[i,0]=math.log(start_prob[i])+math.log(pdf(mean[i],stand_dev[i],y[0]))
        X_pointer[i,0]=0
    for j in range(1,len(y)):
        for i in range(n): 
            maxi=0       
            first=(X[:,j-1])+np.log(transition[:,i])+math.log(pdf(mean[i],stand_dev[i],y[j]))
            X_pointer[i,j]=(first.argmax(axis=0))
            X[i,j]=max(first)
    z=np.ones((len(y)))

    x_result=np.zeros(len(y))

    z[len(y)-1]=X[:,len(y)-1].argmax()
    j=len(y)-1
    while j>=1:
        z[j-1]=X_pointer[int(z[j]),j]
        j=j-1
    result_final=[]
    for i in range(len(y)):
        result_final.append(s[int(z[i])])
    for i in range(len(result_final)):
        if(result_final[i]!=exact_val[i]):
            print(i)
        # else:
        #     print(result_final[i],exact_val[i])
    return result_final
            
    
def stationary_converge(df1,n):
    first_row=[]
    for i in range(n):
        first_row.append(round(df1[0,i],5))
   
    for i in range(1,n):
        for j in range(n):
            if(round(df1[i,j],5)!=first_row[j]):
                return -1
    return 1

def file_write(result_final,bit):
    if bit ==0:
        #  without baum welch er jonyo print
        with open('states_before_bw.txt', 'w') as f:
            for i in range(len(result_final)):
                f.write('"'+result_final[i]+'"')
                f.write('\n')
    else:
        with open('states_after_bw.txt', 'w') as f:
            for i in range(len(result_final)):
                f.write('"'+result_final[i]+'"')
                f.write('\n')
def calculate_stationary(transition,n):
    df=pd.DataFrame(transition)
    result=df
    flag=0
    i=0
    prev_result=result.copy()
    while True:
        result = result.dot(df)
        # print(df,result)
        if(stationary_converge(np.array(result),n)==1):
            break
        i=i+1
    start_prob=[]
    i=0
    # print(result)
    # converge korar por first row ta nibo
    result=result.round(decimals=3)
    return result.iloc[0]

    


            
                



#3################################################################################################### 
params=[]
transition=[]
mean=[]
stand_dev=[]
emission=[]
s=["El Nino","La Nina"]
# parameter reading starts
with open("parameters.txt") as file_in:    
    for line in file_in:
        params.append(line)
# parameter reading stops
#n=number of states
n=int(params[0])
for i in range(n):
    str=params[i+1].split()
    transition.append(string_to_float(str))
#transition matrix given
mean_str=params[len(params)-2].split()
mean=string_to_float(mean_str)
#mean of parameters

stand_dev_str=params[len(params)-1].split()
stand_dev=string_to_float(stand_dev_str)
#stand_dev of parameters
for i in range(len(stand_dev)):
    stand_dev[i]=math.sqrt(stand_dev[i])    


# print(result)
# converge korar por first row ta nibo
# result=result.round(decimals=3)
# ######################################### Generating random input
transition=np.array(transition)
print("Parameters from file")
print(mean,stand_dev,transition)
#comment out while using baum-welch   
for i in range(n):
    mean[i]=random.uniform(1, 300)
    stand_dev[i]=random.uniform(1,15)
for i in range(n):
    sum_roww=0
    for j in range(n):
        transition[i,j]=random.uniform(0.1,0.99999)
        sum_roww+=transition[i,j]
    for j in range(n):
        transition[i,j]=transition[i,j]/sum_roww
print("Parameters after randomization")
print(mean,stand_dev,transition) 
###################################################### 
start_prob=calculate_stationary(transition,n)
# print(start_prob)


start_prob=np.array(start_prob)
start_prob=calculate_stationary(transition,n)



# Baum-Welch implementation starts

transition,mean,stand_dev=baum_welch(transition,y,start_prob,n,mean,stand_dev)
hidden_states=viterbi(n,y,start_prob,transition,mean,stand_dev,viterbi_val_after,s)
print(transition,mean,stand_dev)
start_prob=(calculate_stationary(transition,n))
file_write(hidden_states,1)

# writing learned parameters to file
with open('param_learned.txt', 'w') as f:
  f.write('%d\n' % n)
  for i in range(n):
      for j in range(n):
          f.write("{:.7f}   ".format(transition[i][j]))
      f.write('\n')
  
  for i in range(n):
      f.write("{:.4f}    ".format(mean[i]))
  f.write('\n')
  for i in range(n):
      f.write("{:.6f}    ".format(stand_dev[i]*stand_dev[i]))
  f.write('\n')
  for i in range(n):
      f.write("{:.3f}    ".format(start_prob[i]))
f.close()

  


   




    
    
    




    

