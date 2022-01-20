import pandas as pd

def scale(series):
    return (series - series.min())/(series.max()-series.min())

index_profit = pd.read_excel('指数收益筛选.xlsx')
index_profit.set_index('ETF',inplace=True)
name = pd.read_excel('对照表.xlsx') 
name.set_index('指数',inplace=True)
name_unique = name.drop_duplicates()

index_profit.columns = index_profit.loc['指数',:]
index_profit.drop(['指数','简称'],inplace=True)
index_profit_unique = index_profit.copy()
index_profit_unique = index_profit_unique.T.drop_duplicates().T
index_profit_unique.set_index(pd.to_datetime(index_profit.index),inplace=True)

sub_df = index_profit_unique.copy()
sub_df_scaled = sub_df.apply(scale)
sub_df_scaled_transposed = pd.DataFrame(sub_df_scaled.values.T,index=sub_df_scaled.columns,columns=sub_df_scaled.index)

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(figsize=(20, 16))
Z = linkage(sub_df_scaled_transposed,  method='ward', metric='euclidean')
dendrogram(Z, truncate_mode='level', p=20, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=15,show_contracted=True,labels=name_unique.loc[sub_df_scaled_transposed.index,'简称'].values)
plt.show()
# plt.savefig('分类.jpg')