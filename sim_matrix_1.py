import numpy as np
import tensorflow
import tensorflow as tf

from keras.layers import Dense, Flatten, Conv2D, Embedding, Input, Dot
from pyspark.sql import SparkSession
from tqdm import tqdm

from keras import regularizers


"""
tf2 customized train step and test step
re-define model
test.sim_matrix_posneg_array
取消dense层 直接计算

"""

spark = SparkSession.builder.master("local[2]").appName("SparkByExamples").getOrCreate()
NUM = 100
print("NUM = ", NUM)

input_tar = Input(shape=(1, ))
input_pos = Input(shape=(1, ))
input_neg = Input(shape=(1024,))
embedding_query = Embedding(input_length=None, input_dim=NUM, output_dim=8,
                                         name='embedding_query', trainable=False,
                                         embeddings_regularizer= regularizers.L2(0.01),
                                         embeddings_initializer='random_uniform',
                            )  # shape T,None,8
embedding_search = Embedding( input_length=None, input_dim=NUM, output_dim=8,
                                          name='embedding_search', trainable=True,
                                          embeddings_regularizer= regularizers.L2(0.01),
                                          embeddings_initializer='zeros')  # shape T,None,8
tar = embedding_query(input_tar)  # shape  T,1,8
pos = embedding_search(input_pos)  # shape T,1,8
neg = embedding_search(input_neg)  # shape T,1024,8

p = Dot(axes=2)([tar, pos])  # shape T,1,1
n = Dot(axes=2)([tar, neg])  # shape T,1,1024

pos_part = tf.reduce_sum(tf.reduce_sum(p,  axis=1), axis = 1) # shape T,1

neg_part = tf.reduce_sum(tf.reduce_sum(n,  axis=1), axis = 1) # shape T,1

model_0 = tf.keras.Model(inputs=[input_tar, input_pos, input_neg], outputs=[pos_part, neg_part], name="model_0")
tf.keras.utils.plot_model(model_0, "foo_1.png", show_shapes=True)

# customized train and test step
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_pos_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_neg_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Prepare the metrics.
pos_entropy_metric = tf.keras.metrics.BinaryCrossentropy(from_logits=True, name="pos_entropy_metric")
neg_entropy_metric = tf.keras.metrics.BinaryCrossentropy(from_logits=True, name="neg_entropy_metric")

pos_loss = tf.keras.metrics.Mean(name='pos_loss')
neg_loss = tf.keras.metrics.Mean(name='neg_loss')


model_0.compile(optimizer=optimizer,
                loss=[loss_pos_obj, loss_neg_obj],
                loss_weights = [1., 1.])


#loss_fn = tf.compat.v2.train.AdamOptimizer(0.01).minimize(loss)

@tf.function
def train_step(x, y_true_pos, y_true_neg):
    with tf.GradientTape() as tape:
        pos_part, neg_part = model_0(x, training=True)
        loss_v_pos = loss_pos_obj(y_true_pos, pos_part)
        loss_v_neg = loss_neg_obj(y_true_neg, neg_part)
        # Add any extra losses created during the forward pass.
        #loss_value += sum(model_0.losses)
    grads = tape.gradient([loss_v_pos, loss_v_neg ], model_0.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_0.trainable_variables))

    pos_loss(loss_v_pos)
    neg_loss(loss_v_neg)

    pos_entropy_metric(y_true_pos, pos_part)
    neg_entropy_metric(y_true_neg, neg_part)

# @tf.function
# def test_step(x, y):
#     logits = model_0(x, training=False)
#     loss_value = loss_fn(y, logits)
#
#     test_loss(loss_value)
#     test_entropy_metric(y, logits)


EPOCHS = 0

train_data = spark.sparkContext.parallelize(np.random.randint(10, size=(1024, 3)).tolist(), 8)\
    .map(lambda x: [int(x[0]), [int(x[1])], [int(x[2])]]).repartition(16)\
        .mapPartitions(lambda it: [list(it)]).cache()

# test_data = spark.sparkContext.parallelize(np.random.randint(10, size=(1024, 3)).tolist(), 8)\
#     .map(lambda x: [int(x[0]), int(x[1]), int(x[2])]).repartition(16)\
#         .mapPartitions(lambda it: [list(it)]).cache()

for epoch in range(EPOCHS):
    train_it = train_data.toLocalIterator()
    #test_it = test_data.toLocalIterator()
    #it = zip(train_it, test_it)
    it = train_it
    pos_entropy_metric.reset_states()
    neg_entropy_metric.reset_states()
    pos_loss.reset_states()
    neg_loss.reset_states()

    with tqdm(total=16, desc="process") as pbar:
        i = 0
        for u in tqdm(it, position=0, leave=True):
            i += 1
            pbar.update(1)
            tar = np.array([[t[0]] for t in u])
            pos = np.array([t[1] for t in u])
            neg = np.array([t[2] for t in u])
            # print(np.array(tar).shape)
            # print(np.array(pos).shape)
            # print(np.array(neg).shape)
            pos_labels = np.ones(tar.shape)
            neg_labels = np.zeros(tar.shape)
            train_step([tar, pos, neg], pos_labels, neg_labels)
            #train_step([train[:, :1], train[:, 1:2], train[:, 2:]], labels)

            if i % 4 == 0:
                pbar.set_postfix({'epoch': '{0:4d}'.format(epoch),
                                  'pos_loss': '{0:1.5f}'.format(pos_loss.result()),
                                  'neg_loss': '{0:1.5f}'.format(neg_loss.result()),
                                  'pos_entropy': '{0:1.5f}'.format(pos_entropy_metric.result()),
                                  'neg_entropy': '{0:1.5f}'.format(neg_entropy_metric.result())
                                  })

    pbar.close()

X = np.arange(10, 34).reshape(2, 3, 4)


# embedding_search_result = model_0.get_layer('embedding_search').get_weights()[0]
# print(embedding_search_result.shape, type(embedding_search_result))
#
# # tmp_embedding = model_0.get_layer("embedding_query").get_weights()[0] #shape 100 8
# # tmp_dense = model_0.get_layer("dense_0").get_weights()[0] # 8 32
# # embedding_query_result = np.dot(tmp_embedding, tmp_dense)
# embedding_query_result = model_0.get_layer("embedding_query").get_weights()[0]
# print(embedding_query_result.shape, type(embedding_query_result))
#
# search_df = spark.sparkContext.parallelize(embedding_search_result, 100).map(lambda x: x.tolist()).zipWithIndex()\
#     .toDF(["embedding_search","idx"])
# query_df = spark.sparkContext.parallelize(embedding_query_result, 100).map(lambda x: x.tolist()).zipWithIndex()\
#     .toDF(["embedding_query","idx"])
#
# search_df.join(query_df, ["idx"], "inner").printSchema()

