# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def process_room_feture(csv_file, output):
    """
    处理特征值
    :param csv_file: 原始数据源csv格式
    :param output: 输出的数据集
    """
    result = pd.read_csv(csv_file, header=0, low_memory=False, encoding='utf-8')
    print("原始数据源shape: {}".format(result.shape))
    result.dropna(axis=0, subset=['租金'], inplace=True)
    for name in district_names:
        # 区域进行OneHot编码
        result[name] = 0
        # 基于某一列修改某一列的值
        result.loc[result.district_name == name, name] = 1
    # 计算每个区域的平均房价
    avg_room_rate = result.groupby('district_name')['price_y'].agg(np.mean)
    for name in district_names:
        # 房价为空填充平均值
        result.loc[(result['price_y'].isna() & result[name] == 1), 'price_y'] = avg_room_rate[name]
    result.drop(columns='district_name', inplace=True)
    result['1500内地铁口数量'] = result.apply(lambda x: len(x['1500内地铁口'].split(' ')), axis=1)
    result.drop(columns='1500内地铁口', inplace=True)
    print("输出数据集shape: {}".format(result.shape))
    result.to_csv(output, index=False, encoding="utf-8")
if __name__ == '__main__':
    # uoko房源
    district_names = ['锦江区', '成华区', '高新区', '金牛区', '青羊区', '武侯区', '双流区', '新都区', '郫都区', '温江区', '天府新区', '龙泉驿区']
    process_room_feture('../resources/uoko_cd_rent_room.csv', './uoko_cd_rent_room_dataset.csv')
    # 外部房源
    district_names.append('高新西区')
    district_names.append('天府新区南区')
    process_room_feture('../resources/cd_rent_room.csv', './cd_rent_room_dataset.csv')
