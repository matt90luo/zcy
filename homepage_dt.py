import tensorflow_decision_forests as tfdf
import pandas as pd





# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("/Users/ukiml/penguins.csv")

print(dataset_df.head(3))