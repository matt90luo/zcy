import os
import math
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

print("Found TensorFlow Decision Forests v" + tfdf.__version__)

def split_dataset(dataset, test_ratio=0.30):
    """Splits a panda dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("/Users/feizhu/penguins.csv")

print(dataset_df.count())

print(dataset_df.head(3))

label = "species"
classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)

tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset_df, label="species")

train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model.
model_1.fit(train_ds)

# Export the meta-data to tensorboard.
model_1.make_inspector().export_to_tensorboard("tfdf_tensorboard_logs")


model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

#model_1.save("tfdf_saved_model")

#tfdf.model_plotter.plot_model_in_colab(model_1, tree_idx=0, max_depth=3)


# Remark: The summary's content depends on the learning algorithm
# (e.g. Out-of-bag is only available for Random Forest) and the hyper-parameters
# (e.g. the mean-decrease-in-accuracy variable importance can be disabled in the hyper-parameters).
print(model_1.summary())

# The information in summary are all available programmatically using the model inspector:

# The input features
print(model_1.make_inspector().features())

# The feature importances
print(model_1.make_inspector().variable_importances())

# The model self evaluation is available with the inspector's evaluation():
print(model_1.make_inspector().evaluation())


print(model_1.make_inspector().training_logs())


print(help(tfdf.keras.GradientBoostedTreesModel))


logs = model_1.make_inspector().training_logs()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")
plt.show()

# The previous example did not specify the features, so all the columns were used as input feature (except for the label).
# The following example shows how to specify input features.

feature_1 = tfdf.keras.FeatureUsage(name="year", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
feature_2 = tfdf.keras.FeatureUsage(name="bill_length_mm")
feature_3 = tfdf.keras.FeatureUsage(name="sex")
all_features = [feature_1, feature_2, feature_3]

# Note: This model is only trained with two features. It will not be as good as
# the one trained on all features.

model_2 = tfdf.keras.GradientBoostedTreesModel(
    features=all_features, exclude_non_specified_features=True)

model_2.compile(metrics=["accuracy"])
model_2.fit(train_ds, validation_data=test_ds)

print(model_2.evaluate(test_ds, return_dict=True))


# The hyper-parameter templates of the Gradient Boosted Tree model.
print(tfdf.keras.GradientBoostedTreesModel.predefined_hyperparameters())



#利用keras 做数据预处理 preprocess

body_mass_g = tf.keras.layers.Input(shape=(1,), name="body_mass_g")
body_mass_kg = body_mass_g / 1000.0

bill_length_mm = tf.keras.layers.Input(shape=(1,), name="bill_length_mm")

raw_inputs = {"body_mass_g": body_mass_g, "bill_length_mm": bill_length_mm}
processed_inputs = {"body_mass_kg": body_mass_kg, "bill_length_mm": bill_length_mm}

# "preprocessor" contains the preprocessing logic.
preprocessor = tf.keras.Model(inputs=raw_inputs, outputs=processed_inputs)

# "model_4" contains both the pre-processing logic and the decision forest.
model_4 = tfdf.keras.RandomForestModel(preprocessing=preprocessor)
model_4.fit(train_ds)
print("pre-process")
print(model_4.summary())






