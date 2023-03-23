import keras.backend as K
import pyspark.sql.functions as F

import numpy as np
import tensorflow
import tensorflow as tf

from keras.layers import Dense, Flatten, Conv2D, Embedding, Input, Dot
from pyspark.sql import SparkSession
from tqdm import tqdm

from keras import regularizers
from keras import initializers

"""
使用NCE方法计算softmatrix
"""

spark = SparkSession \
    .builder \
    .appName("pyspark transformSimMatrix") \
    .enableHiveSupport() \
    .getOrCreate()


C_NUM = spark.sql("select c_idx from test.sim_matrix_sample_nce").select(F.max("c_idx")).rdd.map(
    lambda x: int(x[0])).collect()[0] + 1

# W_NUM = spark.sql("select p_idx from test.sim_matrix_sample_nce").select(F.max("p_idx")).rdd.map(
#     lambda x: int(x[0])).collect()[0] + 1


W_NUM = spark.sql("select count(distinct right_video_id) from test.sim_matrix").rdd.map(
    lambda x: int(x[0])).collect()[0]

print("C_NUM = ", C_NUM)
print("W_NUM = ", W_NUM)

input_ctx = Input(shape=(1,))
input_tar = Input(shape=(1024,))

embedding_context = Embedding(input_length=None, input_dim=C_NUM, output_dim=8,
                              name='embedding_context', trainable=False,
                              embeddings_regularizer=regularizers.L2(0.01),
                              embeddings_initializer='random_uniform',
                              )  # shape T,None,8
# context_embedding = tf.Variable(tf.random_uniform([C_NUM, 8], -1.0, 1.0))
d0 = Dense(16, activation='relu',
           kernel_regularizer=regularizers.L2(0.01),
           name="dense_0")

d1 = Dense(32, activation='relu',
           kernel_regularizer=regularizers.L2(0.01),
           name="dense_1")

embedding_word = Embedding(input_length=None, input_dim=W_NUM, output_dim=32,
                           name='embedding_word', trainable=True,
                           embeddings_regularizer=regularizers.L2(0.01),
                           embeddings_initializer='random_uniform')  # shape T,None,32


# nce_weight_word_embedding = tf.Variable(tf.random_uniform([W_NUM, 32], -1.0, 1.0))

print(embedding_word.get_weights())

nce_biases = tf.Variable(tf.zeros([W_NUM]))

cx = embedding_context(input_ctx)  # shape  T,1,8
tar = embedding_word(input_tar)  # shape T,1,32


tmp0 = d0(cx)  # shape T,1,16
tmp1 = tf.squeeze(d1(tmp0), axis = 1) # shape T,1,32

nce = tf.nn.nce_loss(weights = embedding_word.get_weights()[0],
                     biases = nce_biases,
                     labels = input_tar,
                     inputs = tmp1,
                     num_sampled = 4096,
                     num_classes = W_NUM,
                     num_true=1024,
                     sampled_values=None,
                     remove_accidental_hits=False,
                     name="nce_loss")

loss = tf.reduce_mean(nce)

model_0 = tf.keras.Model(inputs=[input_ctx, input_tar], outputs=[tmp1, loss], name="model_0")

# customized train and test step
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

cost_loss = tf.keras.metrics.Mean(name='cost_loss')

# loss_fn = tf.compat.v2.train.AdamOptimizer(0.01).minimize(loss)

@tf.function
def train_train(x):
    with tf.GradientTape() as tape:
        tmp1, loss = model_0(x, training=True)
        # Add any extra losses created during the forward pass.
        # loss_value += sum(model_0.losses)

    grads = tape.gradient(loss, model_0.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_0.trainable_variables))

    cost_loss(loss)
    # pos_loss(loss_v_pos)
    # neg_loss(loss_v_neg)

    # pos_entropy_metric(y_true_pos, pos_part)
    # neg_entropy_metric(y_true_neg, neg_part)


EPOCHS = 0

PARTION_NUM = 1000

train_df = spark.sql("select c_idx, collect_list(p_idx) from test.sim_matrix_sample_nce group by c_idx")

train_data = train_df.rdd \
    .map(lambda x: [int(x[0]), x[1]]).repartition(PARTION_NUM) \
    .mapPartitions(lambda it: [list(it)]).cache()

# test_data = spark.sparkContext.parallelize(np.random.randint(10, size=(1024, 3)).tolist(), 8)\
#     .map(lambda x: [int(x[0]), int(x[1]), int(x[2])]).repartition(16)\
#         .mapPartitions(lambda it: [list(it)]).cache()

for epoch in range(EPOCHS):
    train_it = train_data.toLocalIterator()
    it = train_it
    cost_loss.reset_states()

    with tqdm(total=PARTION_NUM, desc="process") as pbar:
        i = 0
        for u in tqdm(it, position=0, leave=True):
            i += 1
            pbar.update(1)
            cx = np.array([[t[0]] for t in u])
            tar = np.array([t[1] for t in u])
            train_train([cx, tar])
            if i % 10 == 0:
                pbar.set_postfix({'epoch': '{0:4d}'.format(epoch),
                                  'loss': '{0:1.5f}'.format(cost_loss.result())
                                  #   'neg_loss': '{0:1.5f}'.format(neg_loss.result()),
                                  #   'pos_entropy': '{0:1.5f}'.format(pos_entropy_metric.result()),
                                  #   'neg_entropy': '{0:1.5f}'.format(neg_entropy_metric.result())
                                  })

    pbar.close()

# df_video_idx = spark.sql("select left_video_id, left_video_idx from test.sim_matrix_posneg_array").groupBy(
#     "left_video_id", "left_video_idx") \
#     .agg(F.lit(1)).select(F.col("left_video_id").alias("video_id"), F.col("left_video_idx").alias("idx"))
#
# embedding_search_result = model_0.get_layer('embedding_search').get_weights()[0]
# print(embedding_search_result.shape, type(embedding_search_result))


# tmp_embedding = model_0.get_layer("embedding_query").get_weights()[0] #shape 100 8
# tmp_dense = model_0.get_layer("dense_0").get_weights()[0] # 8 32
# embedding_query_result = np.dot(tmp_embedding, tmp_dense)


# do relu
def relu(x):
    return np.maximum(0, x)


# tmp_embedding = model_0.get_layer("embedding_query").get_weights()[0]
# tmp_dense = model_0.get_layer("dense_0").get_weights()[0]
# embedding_query_result = relu(np.dot(tmp_embedding, tmp_dense))
# print(embedding_query_result.shape, type(embedding_query_result))
#
#
# search_df = spark.sparkContext.parallelize(embedding_search_result, 100).map(lambda x: x.tolist()).zipWithIndex() \
#     .toDF(["embedding_search", "idx"])
# query_df = spark.sparkContext.parallelize(embedding_query_result, 100).map(lambda x: x.tolist()).zipWithIndex() \
#     .toDF(["embedding_query", "idx"])
#
# search_df.join(query_df, ["idx"], "inner").join(df_video_idx, ["idx"], "inner").write.mode('overwrite').saveAsTable(
#     "test.sim_matrix_vector_tf_posneg_array_2")
