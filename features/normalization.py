# -*- coding: utf-8 -*-

import pandas as pd


def normalization(dataset, output):
    data = pd.read_csv(dataset, header=0, low_memory=False, encoding='utf-8')

    for col in data.columns:
        if col == '租金':
            continue

        print("开始归一化 {} 特征".format(col))
        # 每列平均值, 标准差
        mean, std = data[col].mean(), data[col].std()
        # z-score 标准化
        data[col] = data.apply(lambda v: (v[col] - mean) / std, axis=1)

    data.to_csv(output, index=False, encoding="utf-8")


if __name__ == '__main__':
    # 外部房源
    normalization('./cd_rent_room_dataset.csv', './cd_rent_room_dataset_normalization.csv')
    # uoko房源
    normalization('./uoko_cd_rent_room_dataset.csv', './uoko_cd_rent_room_dataset_normalization.csv')
