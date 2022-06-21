import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import tensorflow as tf


def join_tuple_string(strings_tuple) -> str:
    return ' '.join(strings_tuple)


# Selected one of the three directory in order to single taking each species
image_dir = Path('/Users/micheletamborrino/Desktop/Fish_Data/images/cropped')
print("Image dir:" + str(image_dir))

# Take all the objects that have anything in the name and ends with .png
filepaths = list(image_dir.glob(r'**/*.png'))

# Uses join_tuple_string that takes as argument the list of the tuples containing the species's name
labels = list(map(join_tuple_string, list(map(lambda x: os.path.split(x)[1].split("_", 2)[:2], filepaths))))
print("I found " + str(len(labels)) + " elements.")

# Creates a sort of column in a table
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

labels.to_csv('labels.csv', index=False)