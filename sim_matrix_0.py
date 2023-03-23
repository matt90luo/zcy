import numpy as np
import tensorflow
import tensorflow as tf

from keras.layers import Dense, Flatten, Conv2D, Embedding, Input, Dot
from pyspark.sql import SparkSession
from tqdm import tqdm

from keras import regularizers


"""
tf2 customized train step and test step
"""

spark = SparkSession.builder.master("local[2]").appName("SparkByExamples").getOrCreate()
NUM = 100
print("NUM = ", NUM)

input_tar = Input(shape=(1, ))
input_pos = Input(shape=(1, ))
input_neg = Input(shape=(1, ))
embedding_query = Embedding(input_length=1, input_dim=NUM, output_dim=8,
                                         name='embedding_query', trainable=True,
                                         embeddings_regularizer= regularizers.L2(0.01),
                                         embeddings_initializer='random_uniform',
                            )  # shape T,1,32
d = Dense(32, activation='relu', name="dense_0" )
embedding_search = Embedding(input_length=1, input_dim=NUM, output_dim=32,
                                          name='embedding_search', trainable=True,
                                          embeddings_regularizer= regularizers.L2(0.01),
                                          embeddings_initializer='zeros')  # shape T,1,32
tar = embedding_query(input_tar)  # shape T,1,8
pos = embedding_search(input_pos)  # shape T,1,32
neg = embedding_search(input_neg)  # shape T,1,32
tmp = d(tar)  # shape T,1,32
p = tf.squeeze(Dot(axes=2)([tmp, pos]), axis=[2])  # shape T,1
n = tf.squeeze(Dot(axes=2)([tmp, neg]), axis=[2])  # shape T,1
logits = p - n

model_0 = tf.keras.Model(inputs=[input_tar, input_pos, input_neg], outputs=logits, name="model_0")
tf.keras.utils.plot_model(model_0, "foo_1.png", show_shapes=True)


# customized train and test step
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Prepare the metrics.
train_entropy_metric = tf.keras.metrics.BinaryCrossentropy(from_logits=True, name="train_entropy_metric")
test_entropy_metric = tf.keras.metrics.BinaryCrossentropy(from_logits=True, name="test_entropy_metric")

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')




@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model_0(x, training=True)
        loss_value = loss_fn(y, logits)
        # Add any extra losses created during the forward pass.
        #loss_value += sum(model_0.losses)
    grads = tape.gradient(loss_value, model_0.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_0.trainable_variables))

    train_loss(loss_value)
    train_entropy_metric(y, logits)

@tf.function
def test_step(x, y):
    logits = model_0(x, training=False)
    loss_value = loss_fn(y, logits)

    test_loss(loss_value)
    test_entropy_metric(y, logits)


EPOCHS = 0

train_data = spark.sparkContext.parallelize(np.random.randint(10, size=(1024, 3)).tolist(), 8)\
    .map(lambda x: [int(x[0]), int(x[1]), int(x[2])]).repartition(16)\
        .mapPartitions(lambda it: [list(it)]).cache()

test_data = spark.sparkContext.parallelize(np.random.randint(10, size=(1024, 3)).tolist(), 8)\
    .map(lambda x: [int(x[0]), int(x[1]), int(x[2])]).repartition(16)\
        .mapPartitions(lambda it: [list(it)]).cache()

for epoch in range(EPOCHS):
    train_it = train_data.toLocalIterator()
    test_it = test_data.toLocalIterator()
    it = zip(train_it, test_it)
    train_entropy_metric.reset_states()
    test_entropy_metric.reset_states()
    train_loss.reset_states()
    test_loss.reset_states()

    with tqdm(total=16, desc="process") as pbar:
        i = 0
        for u in tqdm(it, position=0, leave=True):
            i += 1
            pbar.update(1)
            train = np.array(u[0])
            test = np.array(u[1])
            tar = train[:, :1]
            pos = train[:, 1:2]
            neg = train[:, 2:]
            labels = np.ones(tar.shape)

            train_step([train[:, :1], train[:, 1:2], train[:, 2:]], labels)
            test_step([test[:, :1], test[:, 1:2], test[:, 2:]], labels)

            if i % 4 == 0:
                pbar.set_postfix({'epoch': '{0:4d}'.format(epoch),
                                  'train_loss': '{0:1.5f}'.format(train_loss.result()),
                                  'train_entropy': '{0:1.5f}'.format(train_entropy_metric.result()),
                                  'test_loss': '{0:1.5f}'.format(test_loss.result()),
                                  'test_entropy': '{0:1.5f}'.format(test_entropy_metric.result())
                                  })

    pbar.close()

embedding_search_result = model_0.get_layer('embedding_search').get_weights()[0]
print(embedding_search_result.shape, type(embedding_search_result))


def relu(x):
    return np.maximum(0, x)
# embedding as input then dense finally output
tmp_embedding = model_0.get_layer("embedding_query").get_weights()[0] #shape 100 8
tmp_dense = model_0.get_layer("dense_0").get_weights()[0] # 8 32

embedding_query_result = relu(np.dot(tmp_embedding, tmp_dense))


print(embedding_query_result.shape, type(embedding_query_result))

search_df = spark.sparkContext.parallelize(embedding_search_result, 100).map(lambda x: x.tolist()).zipWithIndex()\
    .toDF(["embedding_search","idx"])
query_df = spark.sparkContext.parallelize(embedding_query_result, 100).map(lambda x: x.tolist()).zipWithIndex()\
    .toDF(["embedding_query","idx"])

search_df.join(query_df, ["idx"], "inner").printSchema()

