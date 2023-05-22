#%% Prep de dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

arquivo = pd.read_csv("/Users/arthurrio/Desktop/Projeto/Modules/Files/arquivo_dummy.csv",delimiter=";")
prod = pd.read_csv("/Users/arthurrio/Desktop/Projeto/Modules/Files/prod_dummy.csv",delimiter=";")
export = pd.read_csv("/Users/arthurrio/Desktop/Projeto/Modules/Files/export.csv",delimiter=",")

#%%
arquivo['Valor_Barril']=arquivo["VALOR_USD"]/arquivo['QUANTIDADE']
arquivo['Valor/KG']=arquivo["VALOR_USD"]/arquivo['KG_LIQUIDO']

arquivo.replace([np.inf, -np.inf], np.nan, inplace=True)
arquivo = arquivo.dropna()

# %%
print(arquivo.info())
print(prod.info())


# %%
X = arquivo["KG_LIQUIDO"].values #atributos preditores
y = arquivo["VALOR_USD"].values #alvo

regressor = LinearRegression()
regressor.fit(X.reshape(-1, 1), y.reshape(-1, 1)) #treinar o modelo de regressão

#%%
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)
plt.scatter(X_treinamento, y_treinamento)
#%%

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X.reshape(-1, 1), y.reshape(-1, 1)) #treinar o modelo de regressão


# %%
intercept = regressor.intercept_
print("intercepto:", intercept)
print()
coef = regressor.coef_
print("coeficiente:",coef)

# %%
predictions = regressor.predict(X.reshape(-1, 1))
predictions
# %%
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

r2 = r2_score(y.reshape(-1, 1), predictions)
mse = mean_squared_error(y.reshape(-1, 1), predictions)


print("R2: ", r2)
print("MSE: ", mse)
#%%

import statsmodels.api as sm

#add constant to predictor variables
X = sm.add_constant(X.reshape(-1, 1))

#fit linear regression model
model = sm.OLS(y.reshape(-1, 1), X).fit()

#view model summary
print(model.summary())


# %%
def prever_valor(rj):
  consumo = intercept + rj * coef
  return consumo

# %%
