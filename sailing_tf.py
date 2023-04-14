from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers
from keras import initializers
from keras import regularizers
from keras.utils.vis_utils import plot_model

GAP = list(map(lambda x: "%03d" % x, range(3, 145, 3))) + list(map(lambda x: "%03d" % x, range(144, 241, 6)))

MODEL_DICT = {}

data_df = pd.read_pickle("data.pkl")
df = pd.read_pickle('station.pkl')

data = data_df.merge(df, on="F_time", how='inner')

def process(d, gap):
    # TF traing
    x = d[d['gap'] == gap]
    x['label'] = x.apply(lambda y: list(map(lambda a: a[1], y.label)), axis=1)
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
        tf.keras.layers.Dense(64, activation='relu', name='d0',
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros(),
                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dense(128, activation='relu', name='d1',
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros(),
                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dense(3, activation=None, name='d2',
                              kernel_initializer=initializers.RandomNormal(stddev=0.01),
                              bias_initializer=initializers.Zeros(),
                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.L2(1e-4))
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    log_dir = "logs/fit/" + gap
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_dataset, validation_data=test_dataset, epochs=128, callbacks=[tensorboard_callback])
    # model.evaluate(test_dataset)
    MODEL_DICT[gap] = model
    prediction = np.expand_dims(MODEL_DICT[gap].predict(test_examples), axis=1)
    station = np.expand_dims(test_labels, axis=1)
    foo_near_station = np.tile(test_examples[:, [[4]]], (1, 1, 3))
    print(np.concatenate((prediction, station, foo_near_station), axis=1))
    return None

for g in GAP:
    process(data, g)

print("finish")
# MODEL_DICT['003'].predict()