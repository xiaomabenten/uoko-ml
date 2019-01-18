import pandas as pd


def divide_data(origin_data, train_data_path, test_data_path):
    # 训练集、测试集以数据集三分之二为分割点
    print('开始切割数据集')
    split_point = (origin_data.shape[0] // 3) * 2
    train_data = origin_data.loc[0:split_point, :]
    test_data = origin_data.loc[split_point + 1:origin_data.shape[0], :]
    train_data.to_csv(train_data_path, index=False, encoding="utf-8")
    test_data.to_csv(test_data_path, index=False, encoding="utf-8")
    print('切割完成')


def normalization_train_data(train_df):
    mean = train_df.mean()
    delta = train_df.std()
    s = (train_df - mean) / delta
    return [s, mean, delta]


def normalization_test_data(test_df, args):
    return (test_df - args[0]) / args[1]


def save_normal_train_data(train_data, path):
    print('开始归一化训练集')
    newDF = pd.DataFrame()
    nomalArgs = {}
    for column in train_data.columns.values.tolist():
        s = []
        if column == '租金':
            # 租金不进行归一化
            s = train_data[column]
        else:
            [s, mean, delta] = normalization_train_data(train_data[column])
            nomalArgs[column] = [mean, delta]
        newDF[column] = s
    newDF.to_csv(path, index=False, encoding="utf-8")
    print('训练集归一化完成')
    return nomalArgs


def save_normal_test_data(test_data, path):
    print('开始归一化测试集')
    newDF_test = pd.DataFrame()
    for column_test in train_data.columns.values.tolist():
        s_test = []
        if column_test == '租金':
            s_test = test_data[column_test]
        else:
            s_test = normalization_test_data(test_data[column_test], nomalArgs[column_test])
        newDF_test[column_test] = s_test
    newDF_test.to_csv(path, index=False, encoding="utf-8")
    print('测试集归一化完成')


def save_args(args, path):
    file = open(path, 'w')
    file.write(str(args))
    file.close()
    print('归一化参数已保存')


if __name__ == '__main__':
    # 原始数据地址
    origin_data = pd.read_csv('./uoko_cd_rent_room_dataset.csv', header=0, low_memory=False, encoding='utf-8')
    # 划分训练集存储地址
    train_data_path = '..//resources//train_data//train_data.csv'
    # 划分测试集存储地址
    test_data_path = '..//resources//test_data//test_data.csv'
    divide_data(origin_data, train_data_path, test_data_path)
    # 读取训练集、测试集
    train_data = pd.read_csv('..//resources//train_data//train_data.csv', header=0, low_memory=False, encoding='utf-8')
    test_data = pd.read_csv('..//resources//test_data//test_data.csv', header=0, low_memory=False, encoding='utf-8')
    # 归一化训练数据
    # 归一化训练集存储地址
    normal_train_path = '..//resources//train_data//uoko_cd_normalization_train.csv'
    nomalArgs = save_normal_train_data(train_data, normal_train_path)
    # 归一化测试数据
    # 归一化测试集存储地址
    normal_test_path = '..//resources//test_data//uoko_cd_normalization_test.csv'
    save_normal_test_data(test_data, normal_test_path)
    # 保存归一化岑书
    path = '..//resources//args//uoko_cd_normalization_args.txt'
    save_args(nomalArgs, path)
