import numpy as np
import tensorflow as tf

from keras.layers import Dense, Flatten, Conv2D, Embedding, Input, Dot
from keras.metrics import binary_accuracy, binary_crossentropy, SparseCategoricalAccuracy, BinaryCrossentropy
from pyspark.sql import SparkSession
from tqdm import tqdm

"""
tf2 use functional API
"""


spark = SparkSession.builder.master("local[2]").appName("SparkByExamples").getOrCreate()
NUM = 10
print("NUM = ", NUM)

input_tar = Input(shape=(1, ))
input_pos = Input(shape=(1, ))
input_neg = Input(shape=(1, ))

embedding_query = Embedding(input_length=1, input_dim=NUM, output_dim=8,
                                         name='embedding_query', trainable=True,
                                         embeddings_initializer='random_uniform')  # shape T,1,32
d = Dense(32, activation='relu')
embedding_search = Embedding(input_length=1, input_dim=NUM, output_dim=32,
                                          name='embedding_search', trainable=True,
                                          embeddings_initializer='zeros')  # shape T,1,32
tar = embedding_query(input_tar)  # shape T,1,8
pos = embedding_search(input_pos)  # shape T,1,32
neg = embedding_search(input_neg)  # shape T,1,32
tmp = d(tar)  # shape T,1,32
p = tf.squeeze(Dot(axes=2)([tmp, pos]), axis=[2])  # shape T,1
n = tf.squeeze(Dot(axes=2)([tmp, neg]), axis=[2])  # shape T,1
logits = p - n

model_0 = tf.keras.Model(inputs=[input_tar, input_pos, input_neg], outputs=logits, name="model_0")
# tf.keras.utils.plot_model(model_0, "foo_2.png", show_shapes=True)

model_0.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["crossentropy"],

)

EPOCHS = 1

train_df = spark.sparkContext.parallelize(np.random.randint(10, size=(1024, 3)).tolist(), 8).toDF()
test_df = spark.sparkContext.parallelize(np.random.randint(10, size=(1024, 3)).tolist(), 8).toDF()

for epoch in range(EPOCHS):
    train_it = train_df.rdd.map(lambda x: [int(x[0]), int(x[1]), int(x[2])]).repartition(16)\
        .mapPartitions(lambda it: [list(it)]).toLocalIterator()

    test_it = test_df.rdd.map(lambda x: [int(x[0]), int(x[1]), int(x[2])]).repartition(16)\
        .mapPartitions(lambda it: [list(it)]).toLocalIterator()

    it = zip(train_it, test_it)

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
            model_0.fit([tar, pos, neg], labels, batch_size=16, epochs=1)
            model_0.evaluate([test[:, :1], test[:,1:2], test[:, 2:]], batch_size=64)
            # train_step(x, labels)
            # if i % 100 == 0:
            #     pbar.set_postfix({'loss': '{0:1.5f}'.format(i), 'epoch': '{0:4d}'.format(epoch),
            #                       'entropy': '{0:1.5f}'.format(train_entropy.result())})

        #pbar.close()
