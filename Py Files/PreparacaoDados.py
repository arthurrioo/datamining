#%%
import pandas as pd 

arquivo=pd.read_csv("/Users/arthurrio/Desktop/Projeto/Modules/Files/EXP_COMPLETA.csv",encoding='utf-8',sep=';')

prod = pd.read_csv("https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/arquivos/ppgn-el/producao-petroleo-m3-1997-2023.csv",delimiter=';')
prod.to_csv("prod.csv", index=False)

export = pd.read_csv("https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/arquivos/ie/petroleo/importacoes-exportacoes-petroleo-2000-2023.csv",delimiter=';')
export.to_csv("export.csv", index=False) 


#%%
#encoding='utf-8'  encoding='ISO-8859-1'
arquivo = arquivo.drop(arquivo[arquivo['CO_NCM']!=27090010].index)

del arquivo['CO_VIA']
del arquivo['CO_URF']
del arquivo['CO_UNID']   
del arquivo['CO_NCM'] 

arquivo=arquivo.rename(columns={'CO_ANO':'ANO',
                                'SG_UF_NCM':"UF_ORIGEM",
                              'CO_MES':'MES',
                              'CO_PAIS':'PAIS_DESTINO',
                              'QT_ESTAT':'QUANTIDADE',
                              'KG_LIQUIDO':'KG_LIQUIDO',
                              'VL_FOB':'VALOR_USD'}) 
arquivo.to_csv("arquivo.csv", index=False)
#%%
arquivo_dummy = pd.get_dummies(arquivo, columns=['UF_ORIGEM'],dtype=float)
prod_dummy = pd.get_dummies(prod, columns=['GRANDE REGIÃO'],dtype=float)

arquivo_dummy=arquivo_dummy.rename(columns={'UF_ORIGEM_BA':'BA',
                                            'UF_ORIGEM_CE':'CE',
                                            'UF_ORIGEM_ES':'ES',
                                            'UF_ORIGEM_MA':'MA',
                                            'UF_ORIGEM_MG':'MG',
                                            'UF_ORIGEM_MN':'MN',
                                            'UF_ORIGEM_ND':'ND',
                                            'UF_ORIGEM_PE':'PE',
                                            'UF_ORIGEM_PR':'PR',
                                            'UF_ORIGEM_RE':'RE',
                                            'UF_ORIGEM_RJ':'RJ',
                                            'UF_ORIGEM_RN':'RN',
                                            'UF_ORIGEM_RS':'RS',
                                            'UF_ORIGEM_SC':'SC',
                                            'UF_ORIGEM_SE':'SE',
                                            'UF_ORIGEM_SP':'SP'}) 

arquivo_dummy.to_csv("arquivo_dummy.csv", index=False)

prod_dummy = prod_dummy.rename(columns={'GRANDE REGIÃO_REGIÃO NORDESTE':'REGIÃO NORDESTE',
                                        'GRANDE REGIÃO_REGIÃO NORTE':'REGIÃO NORTE',
                                        'GRANDE REGIÃO_REGIÃO SUDESTE':'REGIÃO SUDESTE',
                                        'GRANDE REGIÃO_REGIÃO SUL':'REGIÃO SUL'})

prod_dummy.to_csv('prod_dummy.csv',index=False)

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
valor_summary['Minimo valor exportado mensal'] = minimo_mes
valor_summary['Maximo valor exportado mensal'] = maximo_mes


print(valor_summary)
#%%
tabdinamica=x.pivot_table(values=['VALOR_USD'],index=['MES','PAIS_DESTINO'],aggfunc='mean')
tabdinamica=tabdinamica.round(decimals=2)
print(tabdinamica)  
#%%

tabdinamica2=x.pivot_table(values=['VALOR_USD'],index=['UF_ORIGEM','MES'],aggfunc='mean')
tabdinamica2=tabdinamica2.round(decimals=2)
print(tabdinamica2) 

#%%
grafico = pd.DataFrame()
grafico['Minimo valor vendido em cada Mes'] = minimo_mes
grafico['Maximo valor vendido em cada Mes'] = maximo_mes
valor_mes.plot()
grafico.plot()
#%%
#Stabdinamica2.plot.pie()
#%%

estado_soma = pd.DataFrame(x.groupby(x["UF_ORIGEM"])['VALOR_USD'].sum())

# %%
