# %%
from math import isnan
import warnings
import matplotlib.pyplot as plt
from scipy.cluster.vq import *
from scipy.cluster.hierarchy import *
from scipy.optimize import minimize
from typing import Final
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
plt.rcParams['font.sans-serif'] = ['SimHei']


def scale(series):
    return (series - series.min())/(series.max()-series.min())


# %%
IndexList = pd.read_excel('指数表（唯一）.xlsx')
IndexListIndexed = IndexList.set_index('跟踪指数代码')
close = pd.read_excel('close_price.xlsx')
Close = close.set_index('TDATE')
Return = Close / Close.shift(1) - 1

# %%
# Convert Chinese into code
Groups = pd.read_excel('hierarchy_8.xlsx', sheet_name='year_2020')
GroupsSeries = pd.Series(index=np.arange(1, 10))
for i in np.arange(1, 10):
    GroupsSeries[i] = Groups[i].values
GroupsCode = pd.Series(index=GroupsSeries.index)
for i in GroupsCode.index:
    name = GroupsSeries[i]
    index_of_name = [
        a for a in IndexList.index if IndexList.loc[a, '跟踪指数名称'] in GroupsSeries[i]]
    code = IndexList.loc[index_of_name, '跟踪指数代码'].values
    GroupsCode[i] = code
GroupsCode
# %%
Centeroid = pd.Series()
Selected_Index = pd.Series()
start_date = 2020

# Find Centeroid and corresponding code for each group
for i in np.arange(1, 10):
    Return_cut = Return.loc[str(start_date):, GroupsCode[i]]
    temp = Return_cut.dropna(axis=1, thresh=(252*(2020-start_date)))
    temp = temp.dropna(axis=0, how='any')
    temp_copy = temp.copy()
    temp_scaled = temp_copy.apply(scale)
    temp_scaled_transposed = pd.DataFrame(
        temp_scaled.values.T, index=temp_scaled.columns, columns=temp_scaled.index)
    temp_scaled_transposed_whitened = whiten(temp_scaled_transposed)
    codebook = kmeans(temp_scaled_transposed_whitened, 1, True)
    clusters = vq(temp_scaled_transposed_whitened, codebook[0])
    Centeroid[str(i)] = codebook[0][0]
    dist = pd.Series(clusters[1], index=temp_scaled_transposed.index)
    Selected_Index[str(i)] = dist.sort_values().index[0]

name = pd.Series(['大盘红利', '资源', '低碳', '港股', '大杂烩', '债券',
                 '医药', '科技', '消费'], index=np.arange(1, 10))
Index_code = pd.Series(Selected_Index.values, index=np.arange(1, 10))
Index_name = pd.Series(
    IndexListIndexed.loc[Selected_Index, '跟踪指数名称'].values, index=np.arange(1, 10))
output = pd.concat([Index_code, Index_name, name], axis=1)
output.columns = ['指数代码', '指数名称', '板块']
output.to_excel('备选指数.xlsx')


# %%
# equal weight
df1 = Return.loc[:, Selected_Index]
value = df1.apply(np.mean, axis=1)
df1['Mean'] = value
df1_dropna = df1.dropna()
# annulized return
accu_r_df_1 = (df1_dropna + 1).cumprod()
accu_return_1 = accu_r_df_1.iloc[-1] ** (252/accu_r_df_1.shape[0]) - 1
# annulized volatility
accu_vol_1 = df1_dropna.std() * (252**0.5)
# max drawdown
max_drawdown_1 = ((accu_r_df_1.cummax() - accu_r_df_1) /
                  accu_r_df_1.cummax()).max()
# stats
stats_1 = pd.concat([accu_return_1, accu_vol_1, max_drawdown_1], axis=1)
stats_1.columns = ['年化收益率', '年化波动率', '最大回撤']
stats_1.to_excel('等权-汇总.xlsx')
# ax = df1.plot(figsize=(20, 16))
# fig = ax.get_figure()
# fig.savefig('distance_first.jpg')

# %%
# risk parity
# Functions


def calculate_portfolio_var(w, V):
    # 计算组合风险的函数
    w = np.matrix(w)
    return (w*V*w.T)[0, 0]


def calculate_risk_contribution(w, V):
    # 计算单个资产对总体风险贡献度的函数
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w, V))
    # 边际风险贡献
    MRC = V*w.T
    # 风险贡献
    RC = np.multiply(MRC, w.T)/sigma
    return RC


def risk_budget_objective(x, pars):
    # 计算组合风险
    V = pars[0]  # 协方差矩阵
    x_t = pars[1]  # 组合中资产预期风险贡献度的目标向量
    sig_p = np.sqrt(calculate_portfolio_var(x, V))  # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p, x_t))
    asset_RC = calculate_risk_contribution(x, V)
    J = sum(np.square(asset_RC-risk_target.T))[0, 0]  # sum of squared error
    return J


def total_weight_constraint(x):
    return np.sum(x)-1.0


def long_only_constraint(x):
    return x
# 根据资产预期目标风险贡献度来计算各资产的权重


def calcu_w(x):
    w0 = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
#     x_t = [0.25, 0.25, 0.25, 0.25] # 目标是让四个资产风险贡献度相等，即都为25%
    x_t = x
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
            {'type': 'ineq', 'fun': long_only_constraint})
    res = minimize(risk_budget_objective, w0, args=[
                   V, x_t], method='SLSQP', constraints=cons, options={'disp': True})
    w_rb = np.asmatrix(res.x)
    return w_rb


# %%
# calulate weight
# covariance matrix
V = np.matrix(Return.loc[:, Selected_Index].corr())

w_rb = calcu_w([1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9])
print('各指数权重：', w_rb)

r_series = np.matrix(Return.loc[:, Selected_Index])*w_rb.T
df2 = Return.loc[:, Selected_Index]
df2['Mean'] = r_series
df2_dropna = df2.dropna()
# annulized return
accu_r_df = (df2_dropna + 1).cumprod()
accu_return = accu_r_df.iloc[-1] ** (252/accu_r_df.shape[0]) - 1
# annulized volatility
accu_vol = df2_dropna.std() * (252**0.5)
# max drawdown
max_drawdown = ((accu_r_df.cummax() - accu_r_df)/accu_r_df.cummax()).max()
# stats
stats = pd.concat([accu_return, accu_vol, max_drawdown], axis=1)
stats.columns = ['年化收益率', '年化波动率', '最大回撤']
stats.to_excel('风险平价-汇总.xlsx')
