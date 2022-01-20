# %%
import warnings
import matplotlib.pyplot as plt
from scipy.cluster.vq import *
from scipy.cluster.hierarchy import *
from scipy.optimize import *
from typing import Final
import pandas as pd
import numpy as np

# %%
ETFList = pd.read_excel('跟踪ETF.xlsx')
IndexList = pd.read_excel('指数表（唯一）.xlsx')

# %%

close = pd.read_excel('close_price.xlsx')
Close = close.set_index('TDATE')
Return = Close / Close.shift(1) - 1

# %%
# Find ETFs of each index
# ETFList.set_index('ETF_code',inplace=True)
Wanted = IndexList['跟踪指数代码']
ETFFollow = pd.Series()

for name in Wanted:
    ETFFollow[name] = ETFList.loc[(
        ETFList['index_code'] == name), 'ETF_code'].values
ETFFollow

# %%
# Find biggest ETF for each index
ETFFollowBig = pd.Series()
ETFListIndexed = ETFList.set_index('ETF_code')
for index in ETFFollow.index:
    ETFs = ETFFollow[index]
    if(len(ETFs)):
        ETFFollowBig[index] = ETFListIndexed.loc[ETFs,
                                                 'volumn'].sort_values(ascending=False).index.values[0]
    else:
        ETFFollowBig[index] = []
ETFFollowBig

# %%
# convert each group into code
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
# find center ETF for each group
CenterETF = pd.Series(index=GroupsCode.index)
for i in GroupsCode.index:
    ETFs = ETFFollowBig[GroupsCode[i]]
    for etf in ETFs.index:
        if(not len(ETFs[etf])):
            ETFs.drop(etf, inplace=True)
    CenterETF[i] = ETFListIndexed.loc[ETFs,
                                      'volumn'].sort_values(ascending=False).index[0]
CenterETF

# %%
FollowIndex = pd.Series()
for etf in CenterETF:
    FollowIndex[etf] = ETFListIndexed.loc[etf, 'index_code']
IndexListIndexed = IndexList.set_index('跟踪指数代码')
IndexNames = IndexListIndexed.loc[FollowIndex, '跟踪指数名称']

# %%
# 等权
tmp = Return.loc[:, FollowIndex]
value = tmp.apply(np.mean, axis=1)
value_accu = (value + 1).cumprod()
value_accu_dropna = value_accu.dropna()
display((value_accu_dropna[-1] / value_accu_dropna[0])
        ** (252/len(value_accu))-1)
df = ((tmp + 1).cumprod())
df['mean'] = value_accu
IndexNamesList = IndexNames.tolist()
IndexNamesList.append('Mean')
df.columns = IndexNamesList
ax = df.plot(figsize=(20, 16))
fig = ax.get_figure()
fig.savefig('liquidity_first.jpg')

# %%

CenterETF.to_excel('liquidity_first_ETFs.xlsx')
# %%
# 风险平价
# 组合内资产的协方差矩阵（在当前上下文中指策略之间的协方差矩阵）
V = np.matrix([[123, 37.5, 70, 30],
               [37.5, 122, 72, 13.5],
               [70, 72, 321, -32],
               [30, 13.5, -32, 52]])

R = np.matrix([[14], [12], [15], [7]])


def calculate_portfolio_var(w, V):
    '''
    计算投资组合的风险
    :param w: 向量，表示各个资产在投资组合中的权重，
              其实对于这里的输入是一个 1*n 的矩阵
    :param V: 资产之间的协方差矩阵
    :return: 投资组合收益率的方差 sigma^2 （表示投资组合的风险）
    '''
    w = np.matrix(w)
    # w*V*w.T最后是一个1*1的矩阵来着，所以需要取[0,0]
    # w*V*w 是二次型
    return (w*V*w.T)[0, 0]


def calculate_risk_contribution(w, V):
    '''
    计算各个资产对投资组合的风险贡献
    :param w: 向量，表示各个资产在投资组合中的权重，
              其实对于这里的输入是一个 1*n 的矩阵
    :param V: 资产之间的协方差矩阵
    :return:
    '''
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w, V))
    # 边际风险贡献, marginal risk contribution
    # MRC是一个 n*1 的矩阵，代表各个资产的边际风险贡献
    MRC = V*w.T
    # 各个资产对投资组合的风险贡献程度
    RC = np.multiply(MRC, w.T) / sigma
    return RC


def risk_budget_objective(w, params):
    '''
    使用优化求解器求解目标
    :param w: 原始的投资组合中各个资产的权重，是优化器的初始迭代点
    :param params: params[0]代表各资产的协方差矩阵
                   params[1]代表希望各资产对组合风险的贡献程度
    :return:
    '''

    # 计算投资组合风险
    V = params[0]
    expected_rc = params[1]
    sig_p = np.sqrt(calculate_portfolio_var(w, V))
    risk_target = np.asmatrix(np.multiply(sig_p, expected_rc))
    asset_RC = calculate_risk_contribution(w, V)
    J = sum(np.square(asset_RC - risk_target.T))[0, 0]
    return J


def total_weight_constraint(w):
    '''
    在约束求解器中，这个函数的类型是eq, 表示最后返回的这个值要等于0
    :param w:
    :return:
    '''
    return np.sum(w) - 1.0


def long_only_contraint(w):
    # 表示w中的元素都要大于等于0
    return w


def solve_risk_parity_weight(original_w, expected_rc, V):
    '''
    解决风险平价的权重
    :param expected_rc: 期望的
    :param V: 资产间的协方差矩阵
    :return:
    '''
    # original_w = [0.25, 0.25, 0.25, 0.25]
    constraint = ({'type': 'eq',
                   'fun': total_weight_constraint},
                  {'type': 'ineq',
                   'fun': long_only_contraint})
    res = minimize(risk_budget_objective,
                   np.array(original_w),
                   args=[V, expected_rc],
                   method='SLSQP',
                   constraints=constraint,
                   options={'disp': False})

    return np.asmatrix(res.x)


# %%
# Not liquidity
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
plt.rcParams['font.sans-serif'] = ['SimHei']


def scale(series):
    return (series - series.min())/(series.max()-series.min())


# %%
Centeroid = pd.Series()
mark = pd.Series()
mark_alternative = pd.Series()
start_date = 2020

#
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
    mark[str(i)] = dist.sort_values().index[0]
    mark_alternative[str(i)] = dist.sort_values().index[1]

Final_ETF = ETFFollowBig[mark]
Final_ETF_alternative = ETFFollowBig[mark_alternative]

for i, index in enumerate(Final_ETF.index):
    if(not len(Final_ETF[index])):
        Final_ETF.drop(index, inplace=True)
        Final_ETF[Final_ETF_alternative.index[i]] = Final_ETF_alternative[i]

# %%
IndexListIndexed = IndexList.set_index('跟踪指数代码')
Index_selected = IndexListIndexed.loc[Final_ETF.index, '跟踪指数名称']
Index_selected

tmp = Return.loc[:, Index_selected.index]
value = tmp.apply(np.mean, axis=1)
value_accu = (value + 1).cumprod()
value_accu_dropna = value_accu.dropna()
display((value_accu_dropna[-1] / value_accu_dropna[0])
        ** (252/len(value_accu))-1)
df1 = ((tmp + 1).cumprod())
df1['mean'] = value_accu
Index_selected_list = Index_selected.tolist()
Index_selected_list.append('Mean')
df1.columns = Index_selected_list
ax = df1.plot(figsize=(20, 16))
fig = ax.get_figure()
fig.savefig('distance_first.jpg')


# %%
