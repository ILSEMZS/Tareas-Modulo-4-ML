#!/usr/bin/env python
# coding: utf-8

# Queremos mostarar que el algoritmo AdaBoost.M1. es equivalente al algoritmo Forward Stagewise Additive Modeling, 
# si consideramos la función de perdida como se define acontinuación:
# $$
# L(y,f(x))=exp(-y f(x))
# $$

# Considerando el algoritmo forward:
# 1. inicializar $f_{0}(x)=0$
# 
# 2. con m=1 a M
# 
# (a)Calcular $(\beta_{m},\gamma_{m}) = arg min_{\beta,\gamma} \sum^M_1  L(y_{i},f_{m-1}+\beta b(x_{i};\gamma))$ 
# 
# (b)establecer  $f_{m}(x)= f_{m-1}(x) +\beta_{m} b(x;\gamma_{m})$

# Recordemos que AdaBoost.M1. es un algoritmo de clasificacion dicotomica, entonces la función base seria un clasificados $G_{m}$ con soporte en ${1,-1}$
# 
# así en el paso 2.(b) tendríamos que :
# 
# $$(\beta_{m},G_{m})=  arg min_{\beta,\gamma} \sum^M_1 L(y_{i},f_{m-1}+\beta G_{m}(x_{i}))$$
# $$= \sum^M_1 exp(-y_{i} (f_{m-1}+\beta G_{m}(x_{i})))  = \sum^M_1 exp(-y_{i} f_{m-1}-y_{i}\beta G_{m}(x_{i}))$$
# $$= \sum^M_1 exp(-y_{i} f_{m-1})exp(-y_{i}\beta G_{m}(x_{i}))$$

# como $exp(-y_{i} f_{m-1})$ no depende de $\beta ni G$ podemos renombrar el termino como $w_{i}^{(m)}$ y así obtenemos :

# $$\sum^M_1 w_{i}^{(m)}exp(-y_{i}\beta G_{m}(x_{i}))$$

# Ahora analizando el termino $-y_{i}\beta G_{m}(x_{i})$, y recordando que  $-y_{i} , G_{m}(x_{i})$ solo toman valores en ${-1,1}$, reescribimos $-y_{i}\beta G_{m}(x_{i})$ como { $\beta $ si  $y_{i} \ne G_{m}(x_{i})$ y $-\beta$ en el caso contrario }

# Separando nuestra expresión en esos dos casos obtenemos:
# $$exp(-\beta)\sum_{y_{i}=G_{m}(x_{i})} w_{i}^{(m)}) +  exp(\beta)\sum_{y_{i} \ne G_{m}(x_{i})} w_{i}^{(m)})=0$$
# que a su vez se reexprsa como :
# 
# $$
# (\beta_{m},G_{m})=  arg min_{\beta,\gamma} \sum^M_1 L(y_{i},f_{m-1}+\beta G_{m}(x_{i}))
# =arg min_{\beta,\gamma}(exp(-\beta)-exp(\beta))\sum^M_1 I(y_{i} \ne G_{m}(x_{i}))+ exp(-\beta))\sum^M_1 w_{i}^{(m)}
# $$
# 
# De donde :
# 
# $$
# exp(-\beta))\sum_{y_{i}=G_{m}(x_{i})} w_{i}^{(m)}=exp(-\beta)\sum^M_1 w_{i}^{(m)}-exp(-\beta)\sum^M_1 I(y_{i} \ne G_{m}(x_{i}))
# $$
# 

# y finalmente  para minimizar G podemos considerar unicamente :
# 
# $$
# arg min_{\beta,\gamma}(\sum^M_1 w_{i}^{(m)} I(y_{i} \ne G_{m}(x_{i}))
# $$ 
# dado que $exp(-\beta)-exp(\beta)$ y $exp(-\beta))\sum^M_1 w_{i}^{(m)}$ son constantes

# y para minimizar sobre $\beta$ derivando obtenemos:
# 
# $$
# \delta/\delta \beta = (exp(-\beta)-exp(\beta))\sum^M_1 I(y_{i} \ne G_{m}(x_{i}))  w_{i}^{(m)}  -  exp(-\beta))\sum^M_1 w_{i}^{(m)} =0
# $$
# $
# \implies
# $

# $$(exp(-\beta)-exp(\beta))\sum^M_1 I(y_{i} \ne G_{m}(x_{i}))  w_{i}^{(m)}   = exp(-\beta))\sum^M_1 w_{i}^{(m)}
# $$
# $
# \implies
# $
# $$
# (1+exp(2\beta))\sum^M_1 I(y_{i} \ne G_{m}(x_{i}))  w_{i}^{(m)}   = \sum^M_1 w_{i}^{(m)}
# $$
# $
# \implies
# $

# $$
# exp(2\beta)=-1+ \frac{ \sum^M_1 w_{i}^{(m)} }{ \sum^M_1 I(y_{i} \ne G_{m}(x_{i}))  w_{i}^{(m)}  }
# $$

# considerando $err_{m}= \sum^M_1 I(y_{i} \ne G_{m}(x_{i}))  w_{i}^{(m)} $ reescribimos
# 
# $$
# exp(2\beta)=- \frac{ 1-err_{m}}{err_{m}}
# $$

# tomando log de la expresion tenemos
# $$
# 2\beta=log(\frac{ 1-err_{m}}{err_{m}})
# $$
# Pra el paso 2.(b) tendriamos
# $$
#  w_{i}^{(m+1)}=exp(y_{i}f_{m}(x_i))=exp(-y_i f_{m-1}(x_i))exp(-y_i G(x_i)\beta_m)
# $$
# 
# $$
#  w_{i}^{(m+1)}=w_{i}^{(m)}exp(-y_i G(x_i)\beta_m)
# $$
# 
# $$
#  w_{i}^{(m+1)}=w_{i}^{(m)}exp(I(y_{i} \ne G_{m}(x_{i}))exp(\beta_m)
# $$
# 
# La cual es la expresion de 2.(b) como queriamos 

# In[ ]:




