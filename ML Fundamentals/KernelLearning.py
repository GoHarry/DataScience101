# coding: utf-8

# # Kernel Learning
# 
# Let's start looking at a few kernels

# In[1]:

#get_ipython().magic('matplotlib inline')


# In[4]:

import numpy as np
import pylab as pl
import sklearn.metrics.pairwise as pw

figsize=(10, 8)
fig = pl.figure(figsize=figsize)

#%%
# In[6]:

help(pw.polynomial_kernel)


# Let's generate some data and have some look at kernel functions


#%%
# In[15]:

x = pl.linspace(-3, 3, 100).reshape(100,1)


#%%
# In[40]:

def colvec(a):
    return np.array(a).reshape(len(a), 1)


#%%
# In[41]:
pl.figure(figsize=figsize)
pl.plot(x, pw.polynomial_kernel(x, colvec([-2, 0, 2]), coef0=1, degree=3))


#%%
# In[42]:
pl.figure(figsize=figsize)
pl.plot(x, pw.linear_kernel(x, colvec([-2, 0, 2])))


#%%
# In[46]:
pl.figure(figsize=figsize)
pl.plot(x, pw.rbf_kernel(x, colvec([-2, 0, 2]), gamma=1))


# # some 1d classification problem


#%%
# In[90]:

x = np.r_[pl.randn(50, 1) + 3, pl.randn(50, 1) - 3]
y = np.r_[pl.ones(50), -pl.ones(50)]
xp = colvec(np.linspace(-8, 8, 1000))


#%%
# In[88]:
pl.figure(figsize=figsize)
pl.plot(x, colvec(y), 'o'); pl.title("The data!")


#%%
# In[54]:

from sklearn import svm


#%%
# In[95]:

cl = svm.SVC(kernel='linear', C=1)
cl.fit(x,y)


#%%
# In[96]:
pl.figure(figsize=figsize)
pl.plot(x, colvec(y), 'o'); pl.plot(xp, cl.decision_function(xp), '-')


# Try different settings for C to see how regularization affects the location of the hyperplane. Note how for large C, the boundary is really exactly at the value of 1, the margin!

# ## rbf kernel

#%%
# In[107]:

cl = svm.SVC(kernel='rbf', C=1, gamma=1)
cl.fit(x,y)


#%%
# In[108]:
pl.figure(figsize=figsize)
pl.plot(x, colvec(y), 'o')
pl.plot(xp, cl.decision_function(xp), '-')
pl.plot(xp, cl.predict(xp), 'r-')


# Again, try different value for C and rbf to try out different kernel widths

# ## MINST data set

# In[109]:

#%%
from sklearn.datasets import load_digits
digits = load_digits()


# In[119]:

#%%
pl.gray()
i = 3
pl.matshow(digits.images[i])
digits.target[i]


#%%
# careful: range works a bit differently in python 3
pl.figure(figsize=figsize)
for i in range(1, 20):
    pl.subplot(4, 5, i)
    pl.imshow(digits.images[i], interpolation='none')
    pl.title(digits.target[i])
    


#%%

c = svm.SVC()


#%%
# In[130]:

digits.images.shape


#%%

digits.data.shape



#%%
#
def oneVsRest(ds, c):
    i1 = ds.target == c
    i2 = ~i1
    return ds.data[i1 | i2], i1.astype('int') - i2.astype('int')



#%%

oneVsRest(digits, 1)



#%%


x, y = oneVsRest(digits, 1)


#%%


c.fit(x, y)


#%%

c.score(x, y)


#%%

import sklearn.metrics as mt


#%%

mt.zero_one_loss(y, c.predict(x))


#%%
# # Training / Test splits


from sklearn.model_selection import train_test_split


#%%


x, y = oneVsRest(digits, 1)
xt, xe, yt, ye = train_test_split(x, y)


#%%


c = svm.SVC(kernel='poly', C=100, gamma=0.001)
c.fit(xt, yt)
c.score(xe, ye)


#%%
# # Full grid search


from sklearn.model_selection import GridSearchCV


#%%


params = {'kernel': ['rbf'], 'gamma': np.logspace(-3, 3, 7), 'C': np.logspace(-3, 3, 7)}


#%%


gc = GridSearchCV(svm.SVC(), params, verbose=10)


#%%

gc.fit(x, y)


#%%


gc.best_params_


#%%


gc.best_score_



#%%


c = gc.best_estimator_


#%%


c.support_vectors_.shape


#%%


c.dual_coef_.shape


#%%


c.support_vectors_.shape


#%%
# # Support vectors studies


x, y = oneVsRest(digits, 7)


#%%


Cs = np.logspace(-3, 5, 9)
numsv = np.zeros(len(Cs))
for i in range(len(Cs)):
    c = svm.SVC(kernel='rbf', gamma=0.001, C=Cs[i])
    c.fit(x,y)
    numsv[i] = c.support_vectors_.shape[0]

pl.plot(np.log10(Cs), numsv, 'o-')
numsv
    


#%%


def showFit(c, x, y):
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    mx, my = np.meshgrid(np.linspace(mins[0], maxs[0]), np.linspace(mins[1], maxs[1]))
    z = c.decision_function(np.c_[mx.ravel(), my.ravel()])
    pl.contour(mx, my, z.reshape(mx.shape), [-1, 0, 1])
    
    # plot data
    pos = y == 1
    neg = y == -1
    pl.plot(x[pos,0], x[pos,1], 'r+')
    pl.plot(x[neg, 0], x[neg, 1], 'b.')
    pl.jet()
    
    # plot support vectors
   pl.figure(figsize=figsize)
   pl.plot(c.support_vectors_[:,0], c.support_vectors_[:,1], 'ko', fillstyle='none')
   pl.grid()


#%%


import sklearn.datasets as datasets


#%%


x, y = datasets.make_blobs(100, 2, 2); y = 2*y-1


#%%


c = svm.SVC(kernel='linear', C=0.01)
c.fit(x,y)
pl.figure(figsize=figsize)
showFit(c, x, y)





