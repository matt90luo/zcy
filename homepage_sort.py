import os
import math
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


archive_path = tf.keras.utils.get_file("letor.zip",
  "https://download.microsoft.com/download/E/7/E/E7EABEF1-4C7B-4E31-ACE5-73927950ED5E/Letor.zip",
  extract=True)

print(archive_path)

raw_dataset_path = os.path.join(os.path.dirname(archive_path),"OHSUMED/Data/Fold1/trainingset.txt")

def convert_libsvm_to_csv(src_path, dst_path):
  """Converts a libsvm ranking dataset into a flat csv file.

  Note: This code is specific to the LETOR3 dataset.
  """
  dst_handle = open(dst_path, "w")
  first_line = True
  for src_line in open(src_path,"r"):
    # Note: The last 3 items are comments.
    items = src_line.split(" ")[:-3]
    relevance = items[0]
    group = items[1].split(":")[1]
    features = [ item.split(":") for item in items[2:]]

    if first_line:
      # Csv header
      dst_handle.write("relevance,group," + ",".join(["f_" + feature[0] for feature in features]) + "\n")
      first_line = False
    dst_handle.write(relevance + ",g_" + group + "," + (",".join([feature[1] for feature in features])) + "\n")
  dst_handle.close()

# Convert the dataset.
csv_dataset_path="ohsumed.csv"
convert_libsvm_to_csv(raw_dataset_path, csv_dataset_path)

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv(csv_dataset_path)

# Display the first 3 examples.
print(dataset_df.head(3))

print(dataset_df.columns)


dataset_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataset_df, label="relevance", task=tfdf.keras.Task.RANKING)

model = tfdf.keras.GradientBoostedTreesModel(
    task=tfdf.keras.Task.RANKING,
    ranking_group="group",
    num_trees=50)

model.fit(dataset_ds)

logs = model.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.ndcg for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("NDCG (validation)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Loss (validation)")

plt.show()

print(model.summary())

#tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)

# Path to a test dataset using libsvm format.
test_dataset_path = os.path.join(os.path.dirname(archive_path),"OHSUMED/Data/Fold1/testset.txt")
# Convert the dataset.
csv_test_dataset_path="ohsumed_test.csv"
convert_libsvm_to_csv(raw_dataset_path, csv_test_dataset_path)

# Load a dataset into a Pandas Dataframe.
test_dataset_df = pd.read_csv(csv_test_dataset_path)

# Display the first 3 examples.
print(test_dataset_df.head(3))

# Filter by "g_5"
serving_dataset_df = test_dataset_df[test_dataset_df['group'] == 'g_5']
# Remove the columns for group and relevance, not needed for predictions.
serving_dataset_df = serving_dataset_df.drop(['relevance', 'group'], axis=1)

print(serving_dataset_df.count())
# Convert to a Tensorflow dataset
serving_dataset_ds = tfdf.keras.pd_dataframe_to_tf_dataset(serving_dataset_df, task=tfdf.keras.Task.RANKING)

# Run predictions with on all candidate documents
predictions = model.predict(serving_dataset_ds)

serving_dataset_df['prediction_score'] = predictions
print(serving_dataset_df.sort_values(by=['prediction_score'], ascending=False).head(100))





