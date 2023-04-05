import keras.backend as K
# import pyspark.sql.functions as F

import re
import time
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, Embedding, Input, Dot
from pyspark.sql import SparkSession
from tqdm import tqdm
from keras import regularizers
import time

spark = SparkSession.builder
"""
detailpage sort
"""

# spark = SparkSession \
#         .builder \
#         .appName("pyspark sort_transformSimMatrix") \
#         .enableHiveSupport() \
#         .getOrCreate()

input_sample = Input(shape=(32, ), name="input_impression")
input_context = Input(shape=(32, ), name="input_context")
input_label = Input(shape=(1, ))
input_weight = Input(shape=(1, ))


d_0 = Dense(64, activation="relu", name="d_0" )
d_1 = Dense(128, activation="relu", name="d_1" )
d_2 = Dense(1, activation=None, name="d_2")  #logit


class MY_LOSS_LAYER(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MY_LOSS_LAYER, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.weights = self.add_weight(name='weights',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(MY_LOSS_LAYER, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]


tmp_input = tf.keras.layers.Concatenate(axis=1)([input_sample, input_context])

logit = d_2(d_1(d_0(tmp_input))) #shape T,1

cost = tf.reduce_mean(
    tf.squeeze(
        tf.nn.weighted_cross_entropy_with_logits(labels=input_label, logits=logit,
                                                 pos_weight=input_weight)))

model_0 = tf.keras.Model(inputs=[input_sample, input_context], outputs=[logit], name="model_0")

model_1 = tf.keras.Model(inputs=[input_sample, input_context, input_label, input_weight], outputs=[cost], name="model_1")


print("=================== model_0 variable =======================")
print([ x.name for x in model_0.variables])
print("=============================================================\n")

print("=================== model_0 trainable variable =======================")
print([ x.name for x in model_0.trainable_variables])
print("=============================================================\n")

print("=================== model_0 layers =======================")
print([ x.name for x in model_0.layers])
print("=============================================================\n")

print("=================== model_1 variable =======================")
print([ x.name for x in model_1.variables])
print("=============================================================\n")

print("=================== model_1 trainable variable =======================")
print([ x.name for x in model_1.trainable_variables])
print("=============================================================\n")

print("=================== model_1 layers =======================")
print([ x.name for x in model_1.layers])
print("=============================================================\n")


# customized train and test step
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

cost_loss = tf.keras.metrics.Mean(name='cost_loss')

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        cost = model_1(x, training=True)
        # Add any extra losses created during the forward pass.
        #loss_value += sum(model_0.losses)
    grads = tape.gradient(cost, model_0.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_0.trainable_variables))

    cost_loss(cost)

EPOCHS = 10
PARTITION_NUM = 500
train_df = spark.sql("select embedding_query, context, weight, label from test.sort_sample")
# train_data = spark.sparkContext.parallelize(np.random.randint(10, size=(1024, 3)).tolist(), 8)\
#     .map(lambda x: [int(x[0]), [int(x[1])], [int(x[2])]]).repartition(16)\
#         .mapPartitions(lambda it: [list(it)]).cache()

# context 是history embedding
train_data = train_df.rdd.map(lambda x: [x[0], x[1], x[2], x[3]]).repartition(PARTITION_NUM).mapPartitions(lambda it: [list(it)]).cache()

for epoch in range(EPOCHS):
    train_it = train_data.toLocalIterator()
    #test_it = test_data.toLocalIterator()
    #it = zip(train_it, test_it)
    it = train_it
    cost_loss.reset_states()


    with tqdm(total=PARTITION_NUM, desc="process") as pbar:
        i = 0
        for u in tqdm(it, position=0, leave=True):
            i += 1
            pbar.update(1)
            em = np.array([t[0] for t in u])
            cx = np.array([t[1] for t in u])
            weight = np.array([[t[2]] for t in u])
            label = np.array([[t[3]] for t in u])
            train_step([em, cx, label, weight])

            if i % 10 == 0:
                pbar.set_postfix({'epoch': '{0:4d}'.format(epoch),
                                  'cost_loss': '{0:1.5f}'.format(cost_loss.result())
                                  })

    pbar.close()

time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
version = time_stamp + "/"
saved_model_dir = "/home/bigdata/data01/pymlp/tfmodel/"
# model_0.save(saved_model_dir + version)
tf.saved_model.save(model_0, saved_model_dir + version)

