# %%
import warnings
import matplotlib.pyplot as plt
from scipy.cluster.vq import *
from scipy.cluster.hierarchy import *
import pandas as pd
import numpy as np


def scale(series):
    return (series - series.min())/(series.max()-series.min())


reference = pd.read_excel('指数表（唯一）.xlsx')
reference.set_index('跟踪指数代码', inplace=True)

close = pd.read_excel('close_price.xlsx')
Close = close.set_index('TDATE')
Return = Close / Close.shift(1) - 1

# %% [markdown]
# # Hierarchy

# %%
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
plt.rcParams['font.sans-serif'] = ['SimHei']


# sts = np.arange(2015,2021)
sts = np.arange(2020, 2021)
# path = '.\Hierarchy\Hierarchy_stats.xlsx'
# writer = pd.ExcelWriter(path)


# for k in np.arange(2,20):
for k in np.arange(9, 10):
    stats = pd.DataFrame()
    for i, start_date in enumerate(sts):
        Return_cut = Return.loc[str(start_date):, :]
        temp = Return_cut.dropna(axis=1, thresh=(252*(2020-start_date)))
        temp = temp.dropna(axis=0, how='any')
        temp_copy = temp.copy()
        temp_scaled = temp_copy.apply(scale)
        temp_scaled_transposed = pd.DataFrame(
            temp_scaled.values.T, index=temp_scaled.columns, columns=temp_scaled.index)
        Z = linkage(temp_scaled_transposed,  method='ward',
                    metric='euclidean', optimal_ordering=True)
        labels_2 = fcluster(Z, t=k, criterion='maxclust')
        industry = temp_scaled.columns
        belongs = pd.DataFrame(labels_2, columns=[start_date], index=[
                               reference.loc[industry, '跟踪指数名称'], reference.loc[industry, '资产类型']])
        if i == 0:
            stats = belongs
        else:
            stats = pd.concat([stats, belongs], axis=1)
        # plt.figure(figsize=(60, 20))
        # dendrogram(Z, truncate_mode='level', p=15, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=15,show_contracted=True,labels=reference.loc[temp_scaled.columns.values,'跟踪指数名称'].values)
        # plt.show()
    #     path = '.\Hierarchy\Hierarchy-' + str(start_date) + '.jpg'
    #     plt.savefig(path)
    # stats.to_excel(writer, sheet_name=str(k)+'clusters')
# writer.save()

# %% [markdown]
# # Kmeans

# %%
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
plt.rcParams['font.sans-serif'] = ['SimHei']

sts = np.arange(2015, 2021)
belongs = pd.DataFrame()

for i, start_date in enumerate(sts):
    Return_cut = Return.loc[str(start_date):, :]
    temp = Return_cut.dropna(axis=1, thresh=(252*(2020-start_date)))
    temp = temp.dropna(axis=0, how='any')
    temp_copy = temp.copy()
    temp_scaled = temp_copy.apply(scale)
    temp_scaled_transposed = pd.DataFrame(
        temp_scaled.values.T, index=temp_scaled.columns, columns=temp_scaled.index)
    temp_scaled_transposed_whitened = whiten(temp_scaled_transposed)
    codebook = kmeans(temp_scaled_transposed_whitened, 19, True)
    clusters = vq(temp_scaled_transposed_whitened, codebook[0])
    belongs.loc[str(start_date), temp_scaled.columns] = clusters[0]

belongs.columns = [reference.loc[belongs.columns, '跟踪指数名称'].values,
                   reference.loc[belongs.columns, '资产类型'].values]
belongs_transposed = pd.DataFrame(
    belongs.values.T, index=belongs.columns, columns=belongs.index)
belongs_transposed.to_excel('.\Kmeans\Kmeans.xlsx')
