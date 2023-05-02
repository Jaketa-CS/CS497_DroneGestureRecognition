
import pathlib
import cv2 as cv
import numpy as np 
#import pandas as pd
from PIL import Image
import tensorflow as tf 
from keras import Sequential
from keras.layers import Dropout, Dense, Rescaling, Input, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.datasets import mnist
from keras.utils import to_categorical

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
data_dir = pathlib.Path("dataG")
image_count = len(list(data_dir.glob("*/*.jpg")))
print(image_count)

batch_size = 20
img_height = 400
img_width = 500


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(3, 3))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_class = len(class_names)

print(type(train_ds))

# ---------------------build Sequential Model------------------
seq_model = Sequential([
    Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    Conv2D(16, 3, activation="relu", padding="same"),
    MaxPool2D(),
    BatchNormalization(),
    Conv2D(32, 3, padding="same", activation="relu"),
    MaxPool2D(strides =(3,3)),
    BatchNormalization(),
    Dropout(0.5),
    Conv2D(64, 3, padding="same", activation="relu"),
    MaxPool2D(strides =(3,3)),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(units = num_class, activation = "softmax")
])

seq_model.compile(
    optimizer="adam",
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"])

seq_model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints\\checkpt",
                                                 save_weights_only=True,
                                                 verbose=1)

history = seq_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs= 10,
    callbacks=[cp_callback])

seq_model.save("model.h5", include_optimizer=True)

