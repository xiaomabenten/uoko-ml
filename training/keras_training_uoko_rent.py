import keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('..//resources//train_data//uoko_cd_normalization_train.csv', header=0, low_memory=False, encoding='utf-8')
test_data = pd.read_csv('..//resources//test_data//uoko_cd_normalization_test.csv', header=0, low_memory=False, encoding='utf-8')
# 训练样本
train_x = train_data.iloc[:, 1:]
train_y = train_data.iloc[:, 0:1]
# print(train_x.shape[1])
# 测试样本
x_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0:1]
print(train_x.shape)
print(x_test.shape)
print(train_y.shape)
print(y_test.shape)


def build_model(input_shape):
    model = keras.models.Sequential()

    # model.add(keras.layers.Dense(60, activation='relu', input_shape=(input_shape,)))
    # =================================================================
    # data_set_1
    # 80%
    # model.add(keras.layers.Dense(60, activation='relu', input_shape=(input_shape,)))
    # model.add(keras.layers.Dense(100, activation='relu'))
    # model.add(keras.layers.Dense(200, activation='relu'))
    # model.add(keras.layers.Dense(200, activation='relu'))
    # model.add(keras.layers.Dense(100, activation='relu'))
    # model.add(keras.layers.Dense(60, activation='relu'))
    # model.add(keras.layers.Dense(20, activation='relu'))
    # =================================================================
    # =================================================================
    # data_set_1
    # 81.6%  80.7%
    # model.add(keras.layers.Dense(60, activation='relu', input_shape=(input_shape,)))
    # model.add(keras.layers.Dense(100, activation='relu'))
    # model.add(keras.layers.Dense(200, activation='relu'))
    # model.add(keras.layers.Dense(400, activation='relu'))
    # model.add(keras.layers.Dense(400, activation='relu'))
    # model.add(keras.layers.Dense(200, activation='relu'))
    # model.add(keras.layers.Dense(150, activation='relu'))
    # model.add(keras.layers.Dense(60, activation='relu'))
    # model.add(keras.layers.Dense(20, activation='relu'))
    # =================================================================
    # =================================================================
    # data_set_2
    # 87.4%  85.6%
    initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    model.add(keras.layers.Dense(60, activation='relu', input_shape=(input_shape,), kernel_initializer=initializer))
    model.add(keras.layers.Dense(100, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(200, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(400, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(500, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(400, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(200, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(100, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(60, activation='relu', kernel_initializer=initializer))
    model.add(keras.layers.Dense(20, activation='relu', kernel_initializer=initializer))
    # =================================================================
    model.add(keras.layers.Dense(1))
    RMSprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=RMSprop, loss='mse', metrics=['mae'])
    return model


model = build_model(train_x.shape[1])
history = model.fit(train_x, train_y, validation_data=(x_test, y_test), epochs=300, batch_size=10)
scores = model.evaluate(x_test, y_test)
print("平均误差： ", 1 - (scores[1] / train_y.mean()))
print(history.history.keys())

# model.summary()
# test_mse_score, test_mae_score = model.evaluate(x_test, y_test)


# print("test_mse_score:", test_mse_score)
# print("test_mae_score", test_mae_score)

# print("accuracy:", (1-test_mae_score/y_test.mean()))
# print(model.predict(train_x.iloc[-1, :]))
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
