#%% Prep de dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

arquivo = pd.read_csv("https://github.com/arthurrioo/datamining/raw/main/Files/arquivo_dummy.csv",delimiter=";")
prod = pd.read_csv("https://github.com/arthurrioo/datamining/raw/main/Files/prod_dummy.csv",delimiter=";")
export = pd.read_csv("https://github.com/arthurrioo/datamining/raw/main/Files/export.csv",delimiter=",")

arquivo['Valor_Barril']=arquivo["VALOR_USD"]/arquivo['QUANTIDADE']
arquivo['Valor/KG']=arquivo["VALOR_USD"]/arquivo['KG_LIQUIDO']

print(arquivo.info())
print(prod.info())

arquivo.replace([np.inf, -np.inf], np.nan, inplace=True)
data = arquivo.dropna()
#%%
correl_arq = arquivo 
del correl_arq['KEY']
correl_arq = correl_arq.corr()
#%%

# Create a sample dataset
data = {'x1': arquivo["KG_LIQUIDO"].values,
        'x2': arquivo["VALOR_USD"].values,
        'x3': arquivo["QUANTIDADE"].values,
        'x4': arquivo["PAIS_DESTINO"].values,
        'y': arquivo["RJ"].values}
df = pd.DataFrame(data)

# Create the logistic regression model
modelp = LogisticRegression()

# Split the data into input (X) and output (y) variables
Xp = df[['x1', 'x2','x3','x4']]
yp = df['y']

# Fit the model to the data
modelp.fit(Xp, yp)
intercept = modelp.intercept_
print(intercept)

coef = modelp.coef_
print(coef)

# Predict the values
predictions = modelp.predict(Xp)

# Calculate the accuracy
accuracy = accuracy_score(yp, predictions)

# Print the accuracy
print("Accuracy:", accuracy)



#%%
import statsmodels.api as sm
import numpy as np

# Generate some sample data
x1 = np.array(arquivo["KG_LIQUIDO"].values)  # Independent variable 1
x2 = np.array(arquivo["VALOR_USD"].values)  # Independent variable 2
x3 = np.array(arquivo["QUANTIDADE"].values)
x4 = np.array(arquivo["PAIS_DESTINO"].values)
y = np.array(arquivo["RJ"].values)  # Dependent variable (binary)

# Add a constant term to the independent variables
Xs = sm.add_constant(np.column_stack((x1, x2,x3,x4)))

# Fit the logistic regression model
models = sm.Logit(y, Xs)
results = models.fit()

# Print the summary of the regression results
print(results.summary())
#%%


# %%
