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


arquivo['Valor_Barril']=arquivo["VALOR_USD"]/arquivo['QUANTIDADE']
arquivo['Valor/KG']=arquivo["VALOR_USD"]/arquivo['KG_LIQUIDO']

arquivo.replace([np.inf, -np.inf], np.nan, inplace=True)
arquivo = arquivo.dropna()

print(arquivo.info())
print(prod.info())
#%%
correl_arq = arquivo 
del correl_arq['KEY']
correl_arq = correl_arq.corr()

# %%
dados=pd.DataFrame()
dados["RJ"]=arquivo['RJ']
dados["Barril"]=arquivo['Valor_Barril']
dados["KG"]=arquivo['KG_LIQUIDO']
dados["Valor"]=arquivo["VALOR_USD"]
# %%

X = dados.iloc[:, 0:3].values

y = dados.iloc[:, 3].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# %%

# Printando a intercepção
intercept = regressor.intercept_
print(intercept)

coef = regressor.coef_
print(coef)
# %%
predictions = regressor.predict(X)
predictions

# %%
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
r2 = r2_score(y, predictions)
mse = mean_squared_error(y, predictions)
print("R2: ", r2)
print("MSE: ", mse)
#%%
import statsmodels.api as sm

#add constant to predictor variables
X = sm.add_constant(X)

#fit linear regression model
model = sm.OLS(y, X).fit()

#view model summary
print(model.summary())
# %%
