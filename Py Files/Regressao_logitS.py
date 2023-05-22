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
# %%

# Split the data into features (X) and target variable (y)
X = data["KG_LIQUIDO"].values  # Replace 'target_column' with the actual name of the target column
y = data['RJ'].values

corr=np.corrcoef(X,y)[0,1]
corr
#%%

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plt.scatter(X_test, y_test)
#%%
# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_test.reshape(-1, 1), y_test.reshape(-1,1))
#%%
# Make predictions on the testing data
y_pred = model.predict(X_test.reshape(-1,1))

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# %%

import statsmodels.api as sm
import numpy as np

# Generate some sample data
x = X  # Independent variable
y = y  # Dependent variable (binary)

# Add a constant term to the independent variable
X = sm.add_constant(x)

# Fit the logistic regression model
model = sm.Logit(y, X)
results = model.fit()

# Print the summary of the regression results
print(results.summary())

# %%
