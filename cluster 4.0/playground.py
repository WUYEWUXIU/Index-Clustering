# -*- coding: utf-8 -*-
"""
# @Date    : 2021/11/12 12:48 
# @Author  : Jicong Hu
# @Project : 易方达指数基金聚类课题
# @Software: PyCharm
"""


import database_functions_oracle as dbf
import pandas as pd


# formal_excel = pd.read_excel('指数表.xlsx')
# formal_excel.drop_duplicates(subset=['跟踪指数代码']).to_excel('指数表（唯一）.xlsx')

index_map = pd.read_excel('指数表（唯一）.xlsx', index_col=0).reset_index(drop=True)

start_date = '2015-01-01'
end_date = '2021-10-31'

equity_code_list = index_map[index_map['资产类型'] == '股票']['跟踪指数代码'].to_list()
bonds_code_list = index_map[index_map['资产类型'] == '债券']['跟踪指数代码'].to_list()
commodity_code_list = index_map[index_map['资产类型'] == '商品']['跟踪指数代码'].to_list()

equity_close = dbf.wind_index_prices(start_date, end_date, equity_code_list, 'S_DQ_CLOSE', 'AIndexEODPrices')
bonds_close = dbf.wind_index_prices(start_date, end_date, bonds_code_list, 'S_DQ_CLOSE', 'CBIndexEODPrices')
commodity_close = dbf.wind_index_prices(start_date, end_date, commodity_code_list, 'S_DQ_CLOSE', 'CGoldSpotEODPrices')
# equity_os_close = dbf.wind_index_prices(start_date, end_date, equity_code_list, 'S_DQ_CLOSE', 'GlobalIndexEOD')
equity_hk_close = dbf.wind_index_prices(start_date, end_date, equity_code_list, 'S_DQ_CLOSE', 'HKIndexEODPrices')

all_close = pd.concat([equity_close, bonds_close, commodity_close, equity_hk_close], axis=1)

# missing_codes = [x for x in equity_code_list if x not in equity_close.columns]


clustering_outcome = pd.read_excel('./results/Hierarchical聚类结果/Hierarchy_stats.xlsx', '10clusters')


def weave_hierarchy_outcome(cluster_num, method='series'):
    outcome = pd.read_excel('./results/Hierarchical聚类结果/Hierarchy_stats.xlsx', '{}clusters'.format(cluster_num))
    sample_choices = outcome.columns.drop(['跟踪指数名称', '资产类型'])
    cluster_dict = {}
    if method == 'str':
        for choice in sample_choices:
            cluster_dict[choice] = pd.Series(
                [', '.join(outcome[outcome[choice] == x]['跟踪指数名称'].to_list()) for x in outcome[choice].unique()]
            )
    else:
        for choice in sample_choices:
            cluster_dict[choice] = pd.concat(
                [outcome[outcome[choice] == x]['跟踪指数名称'].rename(x).reset_index(drop=True)
                 for x in outcome[choice].unique()], axis=1
            )
    return cluster_dict


def weave_kmeans_outcome(method='series'):
    outcome = pd.read_excel('./results/Kmeans/Kmeans.xlsx')
    sample_choices = outcome.columns.drop(['跟踪指数名称', '资产类型'])
    cluster_dict = {}
    if method == 'str':
        for choice in sample_choices:
            cluster_dict[choice] = pd.Series(
                [', '.join(outcome[outcome[choice] == x]['跟踪指数名称'].to_list()) for x in outcome[choice].unique()]
            )
    else:
        for choice in sample_choices:
            cluster_dict[choice] = pd.concat(
                [outcome[outcome[choice] == x]['跟踪指数名称'].rename(x).reset_index(drop=True)
                 for x in outcome[choice].unique()], axis=1
            )
    return cluster_dict


def write_excel(data_dict, excel_name):
    writer = pd.ExcelWriter(excel_name)
    for name in data_dict:
        data_dict[name].to_excel(writer, 'year_{}'.format(name))
    writer.save()


write_excel(weave_hierarchy_outcome(19, 'str'), 'hierarchy_19_str.xlsx')
write_excel(weave_kmeans_outcome('str'), 'kmeans_str.xlsx')
