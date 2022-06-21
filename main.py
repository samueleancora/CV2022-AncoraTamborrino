import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
#tf.config.experimental.set_memory_growth(gpus[0], True)

def join_tuple_string(strings_tuple) -> str:
    return ' '.join(strings_tuple)


# Selected one of the three directory in order to single taking each species
image_dir = Path('/Users/samueleancora/Downloads/Fish_Data/images/cropped')

# Take all the objects that have anything in the name and ends with .png
filepaths = list(image_dir.glob(r'**/*.png'))

# Uses join_tuple_string that takes as argument the list of the tuples containing the species's name
labels = list(map(join_tuple_string, list(map(lambda x: os.path.split(x)[1].split("_", 2)[:2], filepaths))))

# Creates a sort of column in a table
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenates previous columns providing a data frame.
image_df = pd.concat([filepaths, labels], axis=1)

###############################################################################################
# This may be useful in order to take out from the labeling the images with unwanted names.   #
# Since we have a small number (~40) images called like CUNWCB, we may think to drop them.    #
image_df['Label'] = image_df['Label'].apply(lambda x: np.NaN if x[-3:] == 'png' else x)  #
image_df = image_df.dropna(axis=0)  #
# print(image_df)                                                                             #
###############################################################################################

# We may look at the numbers for each species
print(image_df['Label'].value_counts())

# Create the train and test set
train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

# This process helps us to not run out of memory, loading an image per time
# This is responsible for image augmentation
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# We will use a pre trained model, which is MobileNetV2 transfer CNN model.
pretrained_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                     include_top=False,
                                                     weights='imagenet',
                                                     pooling='avg'
                                                     )
#
pretrained_model.trainable = False

# By doing this we are able to take the first input layer
inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(477, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)
