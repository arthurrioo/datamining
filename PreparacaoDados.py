import pandas as pd 

arquivo=pd.read_csv("EXP_2022.csv",encoding='utf-8',sep=';')
#%%
#encoding='utf-8'  encoding='ISO-8859-1'
arquivo = arquivo.drop(arquivo[arquivo['CO_NCM']!=27090010].index)
arquivo.info()

#%%
del arquivo['SG_UF_NCM']
del arquivo['CO_VIA']
del arquivo['CO_URF']
del arquivo['CO_UNID']    
#%%
arquivo=arquivo.rename(columns={'CO_ANO':'ANO',
                              'CO_MES':'MES',
                              'CO_NCM':'NCM',
                              'CO_PAIS':'PAIS_ORIGEM',
                              'QT_ESTAT':'QUANTIDADE',
                              'KG_LIQUIDO':'KG_LIQUIDO',
                              'VL_FOB':'VALOR_USD'}) 
arquivo.to_csv("arquivo.csv", index=False)


#%%

arquivo.info()
print("O arquivo tem ",arquivo.shape[0]," linhas")
print("O arquivo tem ",arquivo.shape[1]," colunas")
#%%
x = arquivo

def cabecalho(titulo):
    print()
    print("----------------------------------------------------")
    print(titulo)
    print("----------------------------------------------------")
    print()
#%%    
y = x.VALOR_USD.div(10^6).round(decimals=2)

#%%
valor_mes = pd.DataFrame(x.groupby(x["MES"])['VALOR_USD'].sum())
media_mes=pd.DataFrame(x.groupby(x["MES"])['VALOR_USD'].mean())
minimo_mes =pd.DataFrame(x.groupby(x["MES"])['VALOR_USD'].min()) 
maximo_mes = pd.DataFrame(x.groupby(x["MES"])['VALOR_USD'].max())

valor_summary = pd.DataFrame()
valor_summary['Total por mês'] = valor_mes
valor_summary['Media mensal'] = media_mes
valor_summary['Minimo valor vendido em cada Mes'] = minimo_mes
valor_summary['Maximo valor vendido em cada Mes'] = maximo_mes


print(valor_summary)
#%%
tabdinamica=x.pivot_table(values=['VALOR_USD'],index=['MES','PAIS_ORIGEM'],aggfunc='mean')
tabdinamica=tabdinamica.round(decimals=2)
print(tabdinamica) 

#f"Tabela de valores por mês:{valor_summary:, .2f}"
#%%
import matplotlib.pyplot as plt
plt.title(label="Valores em função dos meses")
plt.plot(valor_summary['Total por mês'])
plt.legend(["Soma Mensal"])


