from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers
from keras import initializers
from keras import regularizers
from keras.utils.vis_utils import plot_model

pd.options.display.max_columns = None
pd.options.display.max_rows = None

"to plot easyly, save as DataFrame: Gap, [station]_observing, [station]_prediction, [station]_ecf"

#TARGET_STATIONS = ['K2931', 'K2942', 'K2962']
TARGET_STATIONS = ['K2962']

COLUMNS = [y + "_" + x for x in ['prediction', 'observing', 'ecf'] for y in TARGET_STATIONS]

RES_DF_LIST = []

GAP = list(map(lambda x: "%03d" % x, range(3, 75, 3)))
#+ list(map(lambda x: "%03d" % x, range(144, 241, 6)))

MODEL_DICT = {}

data_df = pd.read_pickle("data.pkl")
df = pd.read_pickle('station.pkl')

data = data_df.merge(df, on="F_time", how='inner')

print(data.head(2))


def process(d, gap):
    # TF traing
    x = d[d['gap'] == gap]
    x['label'] = x.apply(lambda y: [list(map(lambda a: a[1], y.label))[-1]], axis=1)
    x['r'] = np.random.uniform(size=len(x))

    train_examples = np.array(x[x['r'] >= 0.2]['input'].values.tolist())
    test_examples = np.array(x[x['r'] < 0.2]['input'].values.tolist())
    train_labels = np.array(x[x['r'] >= 0.2]['label'].values.tolist())
    test_labels = np.array(x[x['r'] < 0.2]['label'].values.tolist())
    test_time = x[x['r'] < 0.2]['F_time'].values.tolist()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', name='d0',
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros(),
                              kernel_regularizer=regularizers.L2(1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dense(64, activation='relu', name='d1',
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros(),
                              kernel_regularizer=regularizers.L2(1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dense(1, activation=None, name='d2',
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros(),
                              kernel_regularizer=regularizers.L2(1e-4),
                              bias_regularizer=regularizers.L2(1e-4))
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    log_dir = "logs/fit/" + gap
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_dataset, validation_data=test_dataset, epochs=256, callbacks=[tensorboard_callback])
    # model.evaluate(test_dataset)
    MODEL_DICT[gap] = model
    station_prediction = np.expand_dims(MODEL_DICT[gap].predict(test_examples), axis=1)
    station_observing = np.expand_dims(test_labels, axis=1)
    station_ncf_near = np.tile(test_examples[:, [[4]]], (1, 1, 1))

    print(station_prediction.shape)
    print(station_observing.shape)
    print(station_ncf_near.shape)
    print(np.concatenate((station_prediction, station_observing, station_ncf_near), axis=1))

    x = np.concatenate((station_prediction, station_observing, station_ncf_near), axis=1).reshape((-1, 3))
    # print(x.shape)
    # print(x[-4:, :])
    y = [[int(g)]]*x.shape[0]
    df = pd.DataFrame(x, columns=COLUMNS)
    df['gap'] = pd.DataFrame(y, columns=['gap'])
    RES_DF_LIST.append(df)
    return None


for g in GAP:
    process(data, g)

print("finish")

res_df = pd.concat(RES_DF_LIST, ignore_index=True, sort=False)
res_df.to_pickle("plot.pkl")
