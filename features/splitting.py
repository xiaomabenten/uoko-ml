import pandas as pd

# 外部房源
origin_data = pd.read_csv('../features/cd_rent_room_dataset_normalization.csv', header=0, low_memory=False, encoding='utf-8')

# 2/3 训练数据集
train_nums = int(origin_data.shape[0] / 3 * 2)
origin_data.iloc[:train_nums].to_csv('./cd_rent_room_train.csv', index=False, encoding='utf-8')
origin_data.iloc[train_nums:].to_csv('./cd_rent_room_test.csv', index=False, encoding='utf-8')

# uoko房源
origin_data = pd.read_csv('../features/uoko_cd_rent_room_dataset_normalization.csv', header=0, low_memory=False, encoding='utf-8')

# 2/3 训练数据集
train_nums = int(origin_data.shape[0] / 3 * 2)
origin_data.iloc[:train_nums].to_csv('./uoko_cd_rent_room_train.csv', index=False, encoding='utf-8')
origin_data.iloc[train_nums:].to_csv('./uoko_cd_rent_room_test.csv', index=False, encoding='utf-8')
