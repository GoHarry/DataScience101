# -*- coding: utf-8 -*-
"""
Trees, RF and boosting
"""

#%%

import sklearn.datasets as datasets
import pandas as pd
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

#%%
# The default is a CART Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(df, y) #only when we call fit the model actually runs
yhat = dtree.predict(df)
# perfect prediction, but we are predicting the training set (wrong)
yhat - y

#%%

# Visualization
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#%%
# show importances. Look how v3, which is picked the most in the graphic, 
# shows as more important
dtree.feature_importances_

#%%
## show instability
#import numpy as np
#
## TODO fix append y 
#extra_data = pd.DataFrame(np.random.rand(10,4), columns=iris.feature_names)
#extra_y = np.array([2,2,2,2,2,2,2,2,2,2])
#
#
#frames = [df, extra_data]
#after_extra_data = pd.concat(frames)
#after_extra_y = y + extra_y
#
#
#
#
#dtree = DecisionTreeClassifier()
#dtree.fit(after_extra_data, after_extra_y)
#
#dot_data = StringIO()
#export_graphviz(dtree, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())


#%%
# RF example form Andy Mueller's book, p. 85
################################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons



X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
random_state=42)


forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

#%%

# example showing how different tree predictions combine into the RF prediction
import mglearn
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
    
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)


#%%
# breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0, max_features=shape(X_train)[1])
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


# %%

# crossvalidation and booststrap on cancer dataset. Note we didn't do a validation set
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
print("Cross-validation scores:\n{}".format(
cross_val_score(forest, iris.data, iris.target, cv=kfold)))

loo = LeaveOneOut()
scores = cross_val_score(forest, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))



# %%
# artificial dataset with some features being random. See if feature_importances_ 
# catches it, to do after explaining train_test_split
import numpy as np
import matplotlib.pyplot as plt

def ground_truth(x):
    """Ground truth -- function to approximate"""
    return x * np.sin(x) + np.sin(2 * x)

from sklearn.model_selection import train_test_split


def gen_data(n = 100, n_features=10):  
    x = np.random.rand(n, n_features)
    y = np.sin(x[:,0]) + np.sin(2*x[:,1]) + 3*x[:,2] + .4*x[:,3] + x[:,4]
    y_rand = y + 0.75 * np.random.normal(size = n)
    return train_test_split(x, y_rand, test_size=0.5, train_size=0.5)
 

X_train, X_test, y_train, y_test = gen_data(n = 200)
print(X_train.shape)
print(y_train.shape)


#%%

from sklearn.ensemble import RandomForestRegressor
frst = RandomForestRegressor

#%%

# Doing the split by hand
    #y_rand = y + 0.75 * np.random.normal(size = n)
    #train_index = sorted(np.random.choice(n, n/2, replace=False))
    #test_index = [n for n in range(n) if n not in train_index]
    #x_train = x[train_index]
    #y_train = y_rand[train_index]
    #x_test = x[test_index]
    #y_test = y_rand[test_index]
    #return x_train, y_train, x_test, y_test

# %%
#Exercise: Fit a random forest to see if it picked out the informative features

from sklearn.ensemble import RandomForestRegressor

n_features = int(X_train.shape[1]/3) # this is what is recommended, default is auto.

est = RandomForestRegressor(n_estimators=1000, bootstrap=True, criterion='mse', max_depth=None,
           max_features=n_features, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_jobs=-1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

est.fit(X_train, y_train)

rfr2 = RandomForestRegressor(n_estimators=1000)
rfr2.fit(X_train, y_train)


# alternative:
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(rfr2)

#%% antonio
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0, max_features=X_train.shape[1])
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


#%% john
from sklearn.ensemble import RandomForestRegressor
first = RandomForestRegressor(max_features=5, n_estimators=100)
#%%

first.fit(X_train, y_train)
#%%

first.score(X_train, y_train)

first.feature_importances_


#array([0.07043267, 0.0826002 , 0.42570958, 0.06413961, 0.1064666 ,1
#       0.08452344, 0.04692201, 0.03323408, 0.04784364, 0.03812817])

#%%
print("training perf. {:.3f}".format(first.score(X_train, y_train)))
forest.score(X_test, y_test)
print("test perf. {:.3f}".format(first.score(X_train, y_train)))

import matplotlib as plt

def plot_feature_importances_cancer(first):
    n_features = random.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), random.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

#%%
from sklearn.ensemble import RandomForestRegressor

n_features = int(X_train.shape[0]/3) # this is what is recommended, default is auto.

est = RandomForestRegressor(n_estimators=1000, bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_jobs=-1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

est.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(est.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(est.score(X_test, y_test)))


#%%
# boostrapping, adaboost, gradient boosting
################################

# gradient boosting. Play with max_depth and learning_rate

from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0, max_depth=2)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

plot_feature_importances_cancer(gbrt)

