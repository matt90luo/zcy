# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

mnist_path = "/Users/feizhu/Downloads/mnist.npz"

with np.load(mnist_path, allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]

x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)

print(tf.nn.softmax(predictions).numpy())

#定义好损失函数
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#在开始训练之前，使用 Keras Model.compile 配置和编译模型。
#将 optimizer 类设置为 adam，将 loss 设置为您之前定义的 loss_fn 函数，
#并通过将 metrics 参数设置为 accuracy 来指定要为模型评估的指标。
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#使用 Model.fit 方法调整您的模型参数并最小化损失：
model.fit(x_train, y_train, epochs=5)


#Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上检查模型性能。
model.evaluate(x_test,  y_test, verbose=2)


#如果您想让模型返回概率，可以封装经过训练的模型，并将 softmax 附加到该模型
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:2]))
