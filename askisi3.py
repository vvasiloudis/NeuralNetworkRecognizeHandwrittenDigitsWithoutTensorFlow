#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[19]:


digits = pd.read_csv('train.csv')


# In[20]:


digits.shape


# In[21]:


digits.head()


# In[22]:


digits.tail(10)


# In[23]:


digits = np.array(digits)
m, n = digits.shape
np.random.shuffle(digits) # shuffle before splitting into dev and training sets

digits_develop = digits[0:1000].T
Variable_Y_develop = digits_develop[0]
Variable_X_develop = digits_develop[1:n]
Variable_X_develop = Variable_X_develop / 255.

digits_train = digits[1000:m].T
Variable_Y_train = digits_train[0]
Variable_X_train = digits_train[1:n]
Variable_X_train = Variable_X_train / 255.
_,m_train = Variable_X_train.shape


# In[24]:


Variable_Y_train


# In[25]:


def init_params():
    Variable_Function_W1 = np.random.rand(10, 784) - 0.5
    Variable_Function_b1 = np.random.rand(10, 1) - 0.5
    Variable_Function_W2 = np.random.rand(10, 10) - 0.5
    Variable_Function_b2 = np.random.rand(10, 1) - 0.5
    return Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def ForwardProp(Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2, Variable_X):
    Variable_Function_Z1 = Variable_Function_W1.dot(Variable_X) + Variable_Function_b1
    Variable_Function_A1 = ReLU(Variable_Function_Z1)
    Variable_Function_Z2 = Variable_Function_W2.dot(Variable_Function_A1) + Variable_Function_b2
    Variable_Function_A2 = softmax(Variable_Function_Z2)
    return Variable_Function_Z1, Variable_Function_A1, Variable_Function_Z2, Variable_Function_A2

def ReLUDeriv(Z):
    return Z > 0

def OneHot(Variable_Y):
    OneHot_Variable_Y = np.zeros((Variable_Y.size, Variable_Y.max() + 1))
    OneHot_Variable_Y[np.arange(Variable_Y.size), Variable_Y] = 1
    OneHot_Variable_Y = OneHot_Variable_Y.T
    return OneHot_Variable_Y

def BackwardProp(Variable_Function_Z1, Variable_Function_A1, Variable_Function_Z2, Variable_Function_A2, Variable_Function_W1, Variable_Function_W2, Variable_X, Variable_Y):
    OneHot_Variable_Y = OneHot(Variable_Y)
    digits_Z2 = Variable_Function_A2 - OneHot_Variable_Y
    digits_W2 = 1 / m * digits_Z2.dot(Variable_Function_A1.T)
    digits_b2 = 1 / m * np.sum(digits_Z2)
    digits_Z1 = Variable_Function_W2.T.dot(digits_Z2) * ReLUDeriv(Variable_Function_Z1)
    digits_W1 = 1 / m * digits_Z1.dot(Variable_X.T)
    digits_b1 = 1 / m * np.sum(digits_Z1)
    return digits_W1, digits_b1, digits_W2, digits_b2

def UpdateParams(Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2, digits_W1, digits_b1, digits_W2, digits_b2, alpha):
    Variable_Function_W1 = Variable_Function_W1 - alpha * digits_W1
    Variable_Function_b1 = Variable_Function_b1 - alpha * digits_b1    
    Variable_Function_W2 = Variable_Function_W2 - alpha * digits_W2  
    Variable_Function_b2 = Variable_Function_b2 - alpha * digits_b2    
    return Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2


# In[26]:


def get_predictions(Variable_Function_A2):
    return np.argmax(Variable_Function_A2, 0)

def get_accuracy(predictions, Variable_Y):
    print(predictions, Variable_Y)
    return np.sum(predictions == Variable_Y) / Variable_Y.size

def gradient_descent(Variable_X, Variable_Y, alpha, iterations):
    Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2 = init_params()
    for i in range(iterations):
        Variable_Function_Z1, Variable_Function_A1, Variable_Function_Z2, Variable_Function_A2 = ForwardProp(Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2, Variable_X)
        digits_W1, digits_b1, digits_W2, digits_b2 = BackwardProp(Variable_Function_Z1, Variable_Function_A1, Variable_Function_Z2, Variable_Function_A2, Variable_Function_W1, Variable_Function_W2, Variable_X, Variable_Y)
        Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2 = UpdateParams(Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2, digits_W1, digits_b1, digits_W2, digits_b2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(Variable_Function_A2)
            print(get_accuracy(predictions, Variable_Y))
    return Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2


# In[13]:


Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2 = gradient_descent(Variable_X_train, Variable_Y_train, 0.10, 500)


# In[14]:


def MakePredictions(Variable_X, Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2):
    _, _, _, Variable_Function_A2 = ForwardProp(Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2, Variable_X)
    predictions = get_predictions(Variable_Function_A2)
    return predictions

def TestPrediction(index, Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2):
    current_image = Variable_X_train[:, index, None]
    prediction = MakePredictions(Variable_X_train[:, index, None], Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2)
    label = Variable_Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# In[15]:


TestPrediction(0, Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2)
TestPrediction(1, Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2)
TestPrediction(2, Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2)
TestPrediction(3, Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2)


# In[17]:


develop_predictions = MakePredictions(Variable_X_develop, Variable_Function_W1, Variable_Function_b1, Variable_Function_W2, Variable_Function_b2)
get_accuracy(develop_predictions, Variable_Y_develop)


# In[ ]:




