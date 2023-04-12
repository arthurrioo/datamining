import pandas as pd 

arquivo=pd.read_csv("EXP_20_22.csv",encoding='utf-8',sep=';')
#%%
#encoding='utf-8'  encoding='ISO-8859-1'
arquivo = arquivo.drop(arquivo[arquivo['CO_NCM']!=27090010].index)
arquivo.info()

#%%
del arquivo['CO_VIA']
del arquivo['CO_URF']
del arquivo['CO_UNID']    
#%%
arquivo=arquivo.rename(columns={'CO_ANO':'ANO',
                                'SG_UF_NCM':"UF_ORIGEM",
                              'CO_MES':'MES',
                              'CO_NCM':'NCM',
                              'CO_PAIS':'PAIS_DESTINO',
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
tabdinamica=x.pivot_table(values=['VALOR_USD'],index=['MES','PAIS_DESTINO'],aggfunc='mean')
tabdinamica=tabdinamica.round(decimals=2)
print(tabdinamica)  
#%%

tabdinamica2=x.pivot_table(values=['VALOR_USD'],index=['UF_ORIGEM','MES'],aggfunc='mean')
tabdinamica2=tabdinamica2.round(decimals=2)
print(tabdinamica2) 

#f"Tabela de valores por mês:{valor_summary:, .2f}"
#%%
grafico = pd.DataFrame()
grafico['Minimo valor vendido em cada Mes'] = minimo_mes
grafico['Maximo valor vendido em cada Mes'] = maximo_mes
valor_mes.plot()
grafico.plot.density()
grafico.plot()
