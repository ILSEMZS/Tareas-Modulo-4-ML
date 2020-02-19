#!/usr/bin/env python
# coding: utf-8

# Ejercicio
# 
# 
# Entrenando un modelo con PCA
# 
# 
# Considerando xtrain, ytrain definido en las céldas de arriba, crea dos modelos:
# 
# 1.Un pipeline considerando PCA con n_components=0.7 seguido de un modelo KNeighborsClassifier con n_neighbors=5
# 
# 
# 1.KNeighborsClassifier con n_neighbors=5
# 
# 
# Realiza un cross validation con n_components=3.
# 
# 
# ¿Qué resultados arroja cada modelo?
# 
# 
# ¿que ventajas tiene cada modelo?
# 
# ¿Cuánto tarda en correr cada cross-validation?

# In[17]:


from numpy.linalg import svd
# conda install tensorflow
from tensorflow.keras import datasets
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import randn, seed
from scipy.linalg import norm
from sklearn.datasets import make_moons
from ipywidgets import interact, FloatSlider, Dropdown, IntSlider
from sklearn.datasets import make_biclusters, make_moons
from sklearn.neighbors import KNeighborsClassifier


# In[26]:


train, test = datasets.mnist.load_data()
xtrain, ytrain = train

np.random.seed(3141)
indices = np.random.choice(np.arange(len(xtrain)), size=4)
fig, ax = plt.subplots(2, 2)
for axi, ix in zip(ax.ravel(), indices):
    axi.imshow(xtrain[ix])
    axi.axis("off")


# In[27]:


N, M1, M2 = xtrain.shape
X = xtrain.reshape(N, -1)


# In[28]:


get_ipython().run_cell_magic('time', '', 'pipe = Pipeline([\n    ("PCA", PCA(n_components=.7)),\n    ("KNEIGH", KNeighborsClassifier(n_neighbors=5,n_jobs=-1))\n])\nvali_p=cross_validate(pipe, X, ytrain, cv=3, scoring=["accuracy"])')


# In[31]:


vali_p["test_accuracy"].mean()


# In[32]:


get_ipython().run_cell_magic('time', '', 'neigh = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)\nvali=cross_validate(neigh, X, ytrain, cv=3, scoring=["accuracy"])')


# In[33]:


vali["test_accuracy"].mean()


# 1.¿Qué resultados arroja cada modelo? usando accuracy
# 
# modelo 1 : 0.9711666666666666
# 
# modelo 2 : 0.9674166666666667
# 
# 
# 2.¿Qué ventajas tiene cada modelo?
# 
# Dados los resultados obtenidos no podemos encotrar ninguna ventaja del modelo 2 sobre el 1.
# 
# Pues el modelo 1 tiene mejor desempeño y menor tiempo de ejecucción.
# 
# 3.¿Cuánto tarda en correr cada cross-validation?
# 
# modelo 1 : 15.9 s
# 
# modelo 2 : 12min 15s

# In[ ]:




