#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data loading
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np
import random

FC_input=[]
FC_bias=[]
FC_weight=[]


(train_X, train_y), (test_X, test_y) = mnist.load_data()
X_test, X_val, y_test, y_val = train_test_split(test_X, test_y, 
    test_size=0.5, random_state= 1)
print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(X_test.shape))
# print('Y_test:  '  + str(X_val.shape))
train_X[0].shape
# for i in range(9):  
#   pyplot.subplot(330 + 1 + i)
#   pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#   pyplot.show()


# In[2]:


class pool_layer:
    def __init__(self,inputs,pool_size,stride):
        self.inputs = inputs
        self.pool_size= pool_size
        self.stride=stride
#     def pool_back(pool_obj,prev_layer_grad):
#         prev_input=pool_obj.inputs
#         stride=pool_obj.stride
#         pool=pool_obj.pool_size
#         output_height=prev_layer_grad.shape[2]
#         number_channels=prev_layer_grad.shape[1]
#         grad_post=np.zeros(prev_input.shape)
#         for i in range(pool_obj.inputs.shape[0]):
#             for j in range(output_height):
#                 for k in range(output_height):
#                     for l in range(number_channels):
#                           value=prev_input[i,l,j*stride:j*stride+pool,k*stride:k*stride+pool]
#                           value=(value==np.max(value))
#                           grad_post[i,l,j*stride:j*stride+pool,k*stride:k*stride+pool]+=value*prev_layer_grad[i,l,j,k]
#         return grad_post
class con_layer:
    def __init__(self,inputs,stride,pad,weights,bias):
        self.inputs = inputs
        self.stride= stride
        self.pad=pad
        self.weights=weights
        self.bias=bias


# In[3]:


def convolution(filter,image,stride):
  listi=[]
  filter_height=filter.shape[1]
  output_width=int((image.shape[0]-filter_height)/stride)+1
  # print(image,filter,output_width)
  for j in range(output_width):

    temp=[]
    for i in range(output_width):

      portion= image[j*stride:j*stride+filter_height,i*stride:i*stride+filter_height]
      sumi=np.multiply(portion,filter)
      sumi=np.sum(sumi)
      # print(temp_filter[filter_no])
      # print(sumi)
      temp.append(sumi)
    listi.append(temp)
  listi=np.array(listi)
  # print(listi)
  return listi
# temp_filter_1 = np.array([[1,-1,-1], [1,0,0], [1,0,1]])
# temp_image2=np.array([[1,2,2,1,3], [1,1,2,1,-1], [2,2,1,1,0], [0,2,0,2,3], [2,2,1,1,2]])
# (convolution(temp_filter_1,temp_image2))

def relUnit(a):
  result = np.where(a<0, 0, a)
  return result
def pooling(image,pool_size,s):
  listi=[]
  # print(image.shape[0])
  out_size=int((image.shape[0]-pool_size)/s)+1
  for j in range(out_size):
    
    temp=[]
    for i in range(out_size):
      
      portion=image[s*j:s*j+pool_size,s*i:s*i+pool_size]      
      sumi=np.max(portion)
      # print(sumi)      
      temp.append(sumi)

    
    listi.append(temp)
  listi=np.array(listi)
  # print(listi)
  return listi 




# In[4]:


def implement_convo(output_channels,stride,pad,input_image,filter_size):
  first_result=[]
  # /////////bias and filter initialise start
  
#   for i in range(output_channels):
#     p=np.random.randint(-1,1,size=(filter_size,filter_size))
#     p=p*.01
#     temp_filter.append(p)
#   temp_filter=np.array(temp_filter)
  temp_filter=np.random.randn(output_channels,filter_size,filter_size)
  bias=np.random.randn(output_channels)

  # /////////bias and filter initialise end
  # k =number of images
  for k in range(10):
    result_convo=[]
      # i=number of output channels
    for i in range(output_channels):
        # sumation=np.zeros((output_width,output_width))
      # temp_image,bias,temp_filter=convo_init1(pad,input_image,output_channels)
      filter_height=temp_filter.shape[1]
      temp_image=np.pad(input_image[k],((pad,pad),(pad,pad)),mode='constant')
#       print(temp_image.shape)
      output_width=int((temp_image.shape[0]-filter_height)/stride)+1
      p=bias[i]
      t = [ p for i in range(output_width)]
        # print(bias[i])
        
        # print(temp_image.shape)
      t+=convolution(temp_filter[i],temp_image,stride)
        # print(t)
        # print(relUnit(t))
#       print(np.array(t).shape)
      result_convo.append((relUnit(t)))
    result_convo=np.array(result_convo)
    first_result.append(result_convo)
  first_result=np.array(first_result)
#   print(first_result.shape)
  return first_result,temp_filter,bias


# In[5]:


def implement_convo2(output_channels,input_channels,first_result,stride,filter_size):
    filters=[]
#     output_channels=12
    for i in range(output_channels):
      temp_filter=[]
      for j in range(input_channels):
        p=np.random.randint(-1,1,size=(filter_size,filter_size))
        temp_filter.append(p)
      filters.append(temp_filter)
    filters=np.array(filters)
    filters=filters*.01
    # output chanels er soman filter thakbe
    # print(filters2)
#     stride=1
    second_result=[]

    bias=[]
    for i in range(output_channels):
      bias.append(random.randint(-1, 1))
    bias=np.array(bias)
    bias=bias*.01

    # print(bias2)
    for k in range(first_result.shape[0]):
      result_convo2=[]
      for i in range(output_channels):
        # sumation=np.zeros((output_width,output_width))
        temp_image=first_result[k]
        # temp_image=np.pad(temp_image,((0,0)(2,2),(2,2)),mode='constant')
        output_width=int((temp_image.shape[2]-filters[i].shape[2])/stride)+1
        # print(temp_image,filters2[i])
        p=bias[i]
        t = [ p for i in range(output_width)]
        temp_list=[]
        for j in range(temp_image.shape[0]):
          # print(temp_image[j],filters2[i][j])
          t+=(convolution(filters[i][j],temp_image[j],stride))

        result_convo2.append(((relUnit(t))))
      second_result.append(result_convo2)
    second_result=np.array(second_result)
#     print(second_result.shape)
    return second_result,filters,bias


# In[6]:


def pool_implement(a,pool_size,stride):
    output_first_pool=[]
   
    for i in range(a.shape[0]):
        temp=[]
        for j in range(a.shape[1]):
            temp.append(pooling(a[i][j],pool_size,stride))
        output_first_pool.append(temp)
    output_first_pool=np.array(output_first_pool)
    return output_first_pool


# In[7]:


# output_first_conv_rel=[]
output_first_pool=[]
output_second_pool=[]
output_third_pool=[]
stride1=1
pool_size=2
pad=2
filter_size=5
output_first_conv_rel,filter_1,bias_1=implement_convo(6,stride1,pad,train_X,filter_size)
conv1=con_layer(train_X,1,2,filter_1,bias_1)
pool1=pool_layer(output_first_conv_rel,2,1)
# self,inputs,stride,pad,weights,bias
print("After first convolution :",output_first_conv_rel.shape)

output_first_pool=pool_implement(output_first_conv_rel,pool_size,stride1)

print("After pooling :",output_first_pool.shape)


output_second,filters2,bias2=implement_convo2(12,6,output_first_pool,1,filter_size)
print("After second convolution :",output_second.shape)
conv2=con_layer(output_first_pool,1,0,filters2,bias2)
pool2=pool_layer(output_second,2,1)
output_second_pool=pool_implement(output_second,pool_size,stride1)
print("After pooling :",output_second_pool.shape)

output_third,filters3,bias3=implement_convo2(100,output_second.shape[1],output_second_pool,1,filter_size)
print("After third convolution :",output_third.shape)
conv3=con_layer(output_second_pool,1,0,filters3,bias3)


# In[8]:


# ////////////////////////////ARRAY FLattening
flattend_array=[]
for i in range(output_third.shape[0]):
  flattend_array.append(np.reshape(output_third[i],(1,output_third[i].shape[0]*output_third[i].shape[1]*output_third[i].shape[1])))
flattend_array=np.array(flattend_array)
flattend_array.shape


# In[9]:


def FC(shape,flat):
  
  weight=np.random.randn(flat.shape[0],flat.shape[2],shape)
  bias_fc=np.random.randn(flat.shape[0],shape)
  global FC_bias
  FC_bias=bias_fc
  global FC_weight
  FC_weight=weight
  global FC_input
  FC_input=flat
  print(FC_bias.shape)
  result=[]
  for i in range(flat.shape[0]):
    dot_shape=np.dot(flat[i],weight[i])+bias_fc[i]
    result.append(dot_shape)
  result=np.array(result)
  return result

FC_output=[]
# for i in range(flattend_array.shape[0]):
#   FC_output.append()
FC_output=np.array(FC(10,flattend_array))
print(FC_output.shape)
print(FC_weight.shape)


# In[10]:


def softmax(a):
  sumi=0
  a=a.T
  for i in range(a.shape[0]):
    a[i]=np.exp(a[i])
    sumi+=a[i]
  for i in range(a.shape[0]):
    a[i]=a[i]/sumi
  # print(a.T)
  return a
softmax_output=[]
for i in range(FC_output.shape[0]):
    softmax_output.append(softmax(FC_output[i]))
softmax_output=np.array(softmax_output)
print(softmax_output.shape)


# In[11]:


def rel_backward(a):
  result = np.where(a<0, 0, 1)
  return result


# In[12]:


def softmax_backward(y_hat):
    y_real=[]
    for i in range(y_hat.shape[0]):
        values=np.zeros((y_hat.shape[1],1))
        values[train_y[i]]=1
        y_real.append(values)
    y_real=np.array(y_real)
    return 2*(y_hat-y_real)/(y_hat.shape[0])
        
    


# In[13]:


def FC_backward(dz):
    gradient_FC=[]
#     print(dz[0].shape)
#     print(FC_input[0].shape)
#     p=np.dot(dz,FC_input)
#     print(p.shape)
    for i in range(dz.shape[0]):
        gradient_FC.append((np.dot(dz[i],FC_input[i])).T)
    gradient_FC=np.array(gradient_FC)
    global FC_weight
    FC_weight-=.001*gradient_FC
#     print(gradient_FC.shape)
    global FC_bias
    prev_grad=dz[0]
#     print(dz[0])
    for i in range(gradient_FC.shape[2]):
        FC_bias[i]-=.001*prev_grad.T[0][i]
    
    diff_fc=[]
    for i in range(dz.shape[0]):
        diff_fc.append(np.dot(FC_weight[i],dz[i]).T)
    return np.array(diff_fc)
#     for i in range(dz.shape[0]):
#         gradient.append(np.dot(dz[i].T,FC_input[i]))
#     print(np.dot(dz[i],FC_input[i])
#     global FC_weight
#     gradient=np.array(gradient)
#     FC_weight-=.001*gradient
#     print((np.dot(FC_input[i],dz[i].T)))
    


# In[17]:


# def convo_backward(convo_obj,gradient):
#     prev_input=convo_obj.inputs
#     output_gradient=np.zeros(prev_input.shape)
#     updated_weights=np.zeros(convo_obj.weights.shape)
#     updated_bias=np.zeros(convo_obj.bias.shape)
    
#     changed_weights=np.zeros(convo_obj.weights.shape)
#     changed_bias=np.zeros(convo_obj.bias.shape)
    
#     for i in range(prev_input.shape[0]):
#         for j in range(updated_weights.shape[2]):
#             vert_start=j*convo_obj.stride
#             vert_end=j*convo_obj.stride+updated_weights.shape[2]
#             for k in range(updated_weights.shape[2]):
#                 hori_start=k*convo_obj.stride
#                 hori_end=k*convo_obj.stride+updated_weights.shape[2]
#                 for l in range(gradient.shape[1]):
#                     output_gradient[i,:,vert_start:vert_end,hori_start:hori_end]+=convo_obj.weights[:,l,:,:]*gradient[i,l,j,k]
#                     updated_weights[:,l,:,:]+=prev_input[i,:,vert_start:vert_end,hori_start:hori_end]
#                     updated_bias+=gradient[i,l,j,k]
                    
#     changed_weights=convo_obj.weights-.001*updated_weights
#     changed_bias=convo_obj.bias-.001*updated_bias
    
#     return output_gradient,changed_weights,changed_bias
    
    


# In[18]:


# ///////////////////////////BACKWARD PROPAGATION starts


# In[19]:


# softmax er derivative ber korlam
dz_softmax=softmax_backward(softmax_output)

# fully connected er derivative 
fc_backout=FC_backward(dz_softmax)
print(fc_backout.shape)

# flattening layer e reshape koro
reshaped=[]
for i in range(fc_backout.shape[0]):
    reshaped.append(rel_backward(np.reshape(fc_backout[i],(100,18,18))))
#     array reshape with relu backword done
reshaped=np.array(reshaped)
print(reshaped.shape)
# input_check=np.random.randn(10,12,23,23)
# pool_obj=pool_layer(input_check,1,0)
# grad_in=np.random.randn(10,12,22,22)
# grad_pool=pool_back(pool_obj,grad_in)
# print(grad_pool.shape)
input_check=np.random.randn(10,6,27,27)
weights=np.random.randn(12,6,5,5)
bias=np.random.randn(12)
# convo_obj=con_layer(input_check,1,0,weights,bias)
# grad=np.random.randn(10,6,27,27)
# inputs,stride,pad,weights,bias


# In[ ]:





# In[ ]:




