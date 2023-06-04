#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

#%%
arquivo = pd.read_csv("https://github.com/arthurrioo/datamining/raw/main/Files/arquivo.csv",delimiter=",")
prod = pd.read_csv("https://github.com/arthurrioo/datamining/raw/main/Files/prod_dummy.csv",delimiter=";")
export = pd.read_csv("https://github.com/arthurrioo/datamining/raw/main/Files/export.csv",delimiter=",")

arquivo['Valor_Barril']=arquivo["VALOR_USD"]/arquivo['QUANTIDADE']
arquivo['Valor/KG']=arquivo["VALOR_USD"]/arquivo['KG_LIQUIDO']

arquivo.replace([np.inf, -np.inf], np.nan, inplace=True)
arquivo = arquivo.dropna()

# %%
arquivox = pd.DataFrame()
arquivox['Valor/KG']=arquivo['Valor/KG']
arquivox['VALOR_USD']=arquivo['VALOR_USD']
arquivox['Valor_Barril']=arquivo['Valor_Barril']
arquivox.info()

#%%
# Gerando o dendrograma 

## Inicialmente, vamos utilizar: 
## Distancia euclidiana e metodo de encadeamento single linkgage


# Opcoes para o metodo de encadeamento ("method"):
    ## single
    ## complete
    ## average

# Opcoes para as distancias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

plt.figure(figsize=(10,5))

dendrogram = sch.dendrogram(sch.linkage(arquivox, method = 'average', metric = 'canberra'), no_labels=True)
plt.title('Dendrograma', fontsize=16)
plt.ylabel('Distancia Euclidiana', fontsize=16)
plt.xlabel('x', fontsize=16)
plt.show()

# %%
cluster_sing = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'single')
indica_cluster_sing = cluster_sing.fit_predict(arquivox)
arquivo['cluster_single'] = indica_cluster_sing
arquivo
#%%
# Comparando clusters resultantes por diferentes metodos de encadeamento

# Complete linkage

cluster_comp = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')
indica_cluster_comp = cluster_comp.fit_predict(arquivox)

arquivo['cluster_complete'] = indica_cluster_comp
arquivo

# Comparando clusters resultantes por diferentes mÃ©todos de encadeamento
#%%
# Average linkage

cluster_avg = AgglomerativeClustering(n_clusters = 4, affinity = 'canberra', linkage = 'average')
indica_cluster_avg = cluster_avg.fit_predict(arquivox)

print(indica_cluster_avg, "\n")

arquivo['cluster_average'] = indica_cluster_avg
arquivo

#%%
#Cluster Nao Hierarquico K-means

# Considerando que identificamos 3 possiveis clusters na analise hierarquica

kmeans = KMeans(n_clusters = 3, init = 'random')

# Execução do algoritmo kmeans
pred_y = kmeans.fit_predict(arquivox)

# Para identificarmos os clusters gerados
kmeans_clusters = kmeans.labels_
kmeans_clusters

arquivo['cluster_kmeans'] = kmeans_clusters
arquivo


# %%
plt.ylim(0, 1000) #range do eixo y
plt.scatter(arquivox.iloc[:,1], arquivox.iloc[:,2], c = pred_y) #posicionamento dos eixos x e y
plt.grid() #função que desenha a grade no nosso gráfico
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], s = 70, c = 'red') #posição de cada centroide no gráfico
plt.show()
# %%
# Identificando as coordenadas centroides dos clusters finais

cent_finais = pd.DataFrame(kmeans.cluster_centers_)
cent_finais.columns = arquivox.columns
cent_finais.index.name = 'cluster'
cent_finais
# %%

distancias = {}

for k in range(1, 21):
  kmeans_ = KMeans(n_clusters=k)
  kmeans_.fit(arquivox)
  distancias[k] = kmeans_.inertia_

# %%
sns.pointplot(x = list(distancias.keys()), y = list(distancias.values()))
plt.axhline(y = (1.5*10**19), color = 'red', linestyle = '--')
plt.show()

# %%

# Analise de variancia de um fator:
# As variaveis que mais contribuem para a formacao de pelo menos um dos clusters

def teste_f_kmeans(kmeans, dataframe):
    
    variaveis = dataframe.columns

    centroides = pd.DataFrame(kmeans.cluster_centers_)
    centroides.columns = dataframe.columns
    centroides
    
    print("Centroides: \n", centroides ,"\n")

    df = dataframe[variaveis]

    unique, counts = np.unique(kmeans.labels_, return_counts=True)

    dic = dict(zip(unique, counts))

    qnt_clusters = kmeans.n_clusters

    observacoes = len(kmeans.labels_)

    df['cluster'] = kmeans.labels_

    output = []

    for variavel in variaveis:

        dic_var={'variavel':variavel}

        # variabilidade entre os grupos

        variabilidade_entre_grupos = np.sum([dic[index]*np.square(observacao - df[variavel].mean()) for index, observacao in enumerate(centroides[variavel])])/(qnt_clusters - 1)

        dic_var['variabilidade_entre_grupos'] = variabilidade_entre_grupos

        variabilidade_dentro_dos_grupos = 0

        for grupo in unique:

            grupo = df[df.cluster == grupo]

            variabilidade_dentro_dos_grupos += np.sum([np.square(observacao - grupo[variavel].mean()) for observacao in grupo[variavel]])/(observacoes - qnt_clusters)

        dic_var['variabilidade_dentro_dos_grupos'] = variabilidade_dentro_dos_grupos

        dic_var['F'] =  dic_var['variabilidade_entre_grupos']/dic_var['variabilidade_dentro_dos_grupos']
        
        dic_var['sig F'] =  1 - stats.f.cdf(dic_var['F'], qnt_clusters - 1, observacoes - qnt_clusters)

        output.append(dic_var)

    df = pd.DataFrame(output)
    
    print(df)

    return df

# Os valores da estatÃ­stica F sÃ£o bastante sensÃ­veis ao tamanho da amostra

output = teste_f_kmeans(kmeans,arquivox)

# %%

## Elaborado com base na "inercia": distancia de cada obervobservacao para o centroide de seu cluster
## Quanto mais proximos entre si e do centroide, menor a inercia
# Normalmente, busca-se o "cotovelo", ou seja, o ponto onde a curva "dobra"

inercias = []
K = range(1,arquivox.shape[0])
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(arquivox)
    inercias.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, inercias, 'bx-')
plt.axhline(y = 2.2, color = 'red', linestyle = '--')
plt.xlabel('Num Clusters', fontsize=16)
plt.ylabel('Inercias', fontsize=16)
plt.title('Metodo do Elbow', fontsize=16)
plt.show()

# %%
