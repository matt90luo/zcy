import keras.backend as K
# import pyspark.sql.functions as F
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, Flatten, Conv2D, Embedding, Input, Dot
# from pyspark.sql import SparkSession
from tqdm import tqdm

from keras import regularizers
from keras import initializers

"""
NCE
"""

SIM_MATRIX_SMAPLE_NCE = "test.sim_matrix_sample_nce"
SIM_MATRIX_MAPPING_NCE ="test.sim_matrix_mapping_nce"
SIM_MATRIX_EMBEDDING_RESULT = "test.sim_matrix_embedding_nce"

# spark = SparkSession \
#     .builder \
#     .appName("pyspark transformSimMatrix") \
#     .enableHiveSupport() \
#     .getOrCreate()

C_NUM = 128
W_NUM = 256

class MY_NCE_LOSS(keras.layers.Layer):
    def __init__(self, input_dim=W_NUM, output_dim=31):
        super(MY_NCE_LOSS, self).__init__()
        w_init = tf.zeros_initializer()
        #w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
            name="nce_weights"
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(input_dim,), dtype="float32"),
            trainable=True,
            name="nce_bias"
        )

    def call(self, inputs, labels):
        return tf.nn.nce_loss(weights = self.w,
                     biases = self.b,
                     labels = labels,
                     inputs = inputs,
                     num_sampled = 16384,
                     num_classes = W_NUM,
                     num_true=1024,
                     sampled_values=None,
                     remove_accidental_hits=False,
                     name="nce_loss")



# C_NUM = spark.sql("select c_idx from test.sim_matrix_sample_nce").select(F.max("c_idx")).rdd.map(
#     lambda x: int(x[0])).collect()[0] + 1

#W_NUM = spark.sql("select p_idx from test.sim_matrix_sample_nce").select(F.max("p_idx")).rdd.map(lambda x: int(x[0])).collect()[0] + 1
#W_NUM = spark.sql("select count(distinct right_video_id) from test.sim_matrix").rdd.map(lambda x: int(x[0])).collect()[0]

print("C_NUM = ", C_NUM)
print("W_NUM = ", W_NUM)

input_ctx = Input(shape=(1,))
input_tar = Input(shape=(1024,))

embedding_context = Embedding(input_length=None, input_dim=C_NUM, output_dim=8,
                              name='embedding_context', trainable=True,
                              embeddings_regularizer=regularizers.L2(0.01),
                              embeddings_initializer='random_uniform',
                              )  # shape T,None,8
# context_embedding = tf.Variable(tf.random_uniform([C_NUM, 8], -1.0, 1.0))
d0 = Dense(16, activation='relu',
           kernel_regularizer=regularizers.L2(0.01),
           name="dense_0")

d1 = Dense(31, activation='relu',
           kernel_regularizer=regularizers.L2(0.01),
           name="dense_1")

cx = embedding_context(input_ctx)  # shape  T,1,8
tmp0 = d0(cx)  # shape T,1,16
squeeze = tf.keras.layers.Reshape(target_shape=(31,), input_shape=(1, 31))
tmp1 = squeeze(d1(tmp0)) # shape T,31

nce_layer = MY_NCE_LOSS(W_NUM, 31)
nce = nce_layer(tmp1, input_tar)
loss = tf.reduce_mean(nce)

model_0 = tf.keras.Model(inputs=[input_ctx, input_tar], outputs=[loss], name="model_0")
tf.keras.utils.plot_model(model_0, "foo_1.png", show_shapes=True)

print([x.name for x in model_0.trainable_variables])

print([x.name for x in model_0.layers])

print(model_0.get_layer("my_nce_loss").weights[0])

#print(model_0.trainable_variables)
#print(embedding_word.get_weights()[0])


# customized train and test step
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

cost_loss = tf.keras.metrics.Mean(name='cost_loss')

# loss_fn = tf.compat.v2.train.AdamOptimizer(0.01).minimize(loss)

@tf.function
def train_train(x):
    with tf.GradientTape() as tape:
        loss = model_0(x, training=True)
        # Add any extra losses created during the forward pass.
        # loss_value += sum(model_0.losses)

    grads = tape.gradient(loss, model_0.trainable_variables + embedding_word.trainable_variables + nce_biases.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_0.trainable_variables + embedding_word.trainable_variables + nce_biases.trainable_variables))
    cost_loss(loss)

EPOCHS = 1

PARTION_NUM = 1000

train_df = spark.sql("select c_idx, collect_list(p_idx) from test.sim_matrix_sample_nce group by c_idx")

train_data = train_df.rdd \
    .map(lambda x: [x[0], x[1]]).repartition(PARTION_NUM) \
    .mapPartitions(lambda it: [list(it)]).cache()


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
                                  })

    pbar.close()

#严重怀疑是因为传值还是传引用导致了这个问题
print(model_0.get_layer("embedding_word").get_weights()[0])
print(nce_biases.get_weights()[0])


# df_video_search_idx = spark.sql(f"select left_video_id, c_idx from {SIM_MATRIX_SMAPLE_NCE}").groupBy("left_video_id", "c_idx")\
#     .agg(F.lit(1)).select(F.col("left_video_id").alias("video_id"), F.col("c_idx").alias("idx"))
#
# #这里会有点问题，anyway topN right_video_id is just a sub set of ALL videos!
# # df_video_query_idx = spark.sql("select p_right_video_id, p_idx from test.sim_matrix_sample_nce").groupBy("p_right_video_id", "p_idx")\
# #     .agg(F.lit(1)).select(F.col("p_right_video_id").alias("video_id"), F.col("p_idx").alias("idx"))
# df_video_query_idx = spark.sql(f"select video_id, p_idx from {SIM_MATRIX_MAPPING_NCE}")\
#     .select(F.col("video_id"), F.col("p_idx").alias("idx"))
#
# embedding_query_result = tf.concat([embedding_word.get_weights()[0], tf.expand_dims(nce_biases, axis=1)], 1).numpy()
# print(embedding_query_result.shape, type(embedding_query_result))


tmp_embedding = model_0.get_layer("embedding_context").get_weights()[0]
print("embedding weights length= ", len(model_0.get_layer("embedding_context").get_weights()))
dense_0 = model_0.get_layer("dense_0")
print("dense weights length = ", len(dense_0.get_weights()))
dense_1 = model_0.get_layer("dense_1")
embedding_search_result_tmp = dense_1(dense_0(tmp_embedding))
print(embedding_search_result_tmp.shape, type(embedding_search_result_tmp))
embedding_search_result = tf.concat([tf.convert_to_tensor(embedding_search_result_tmp), tf.ones([C_NUM, 1])], axis=1).numpy()

print(embedding_search_result.shape, type(embedding_search_result))

search_df = spark.sparkContext.parallelize(embedding_search_result, 100).map(lambda x: x.tolist()).zipWithIndex()\
    .toDF(["embedding_search","idx"])
query_df = spark.sparkContext.parallelize(embedding_query_result, 100).map(lambda x: x.tolist()).zipWithIndex()\
    .toDF(["embedding_query","idx"])
a = search_df.join(df_video_search_idx, ["idx"], "inner").select("video_id", "embedding_search")
b = query_df.join(df_video_query_idx, ["idx"], "inner").select("video_id", "embedding_query")
a.join(b, ["video_id"], "inner").write.mode('overwrite').saveAsTable(SIM_MATRIX_EMBEDDING_RESULT)

# # dense 有bias


embedding_search_result = tf.concat([embedding_search_result_tmp, tf.ones([C_NUM, 1])], axis=1).numpy()
embedding_query_result = tf.concat([embedding_word.get_weights()[0], tf.expand_dims(nce_biases, axis=1)], 1).numpy()




