#! /usr/bin/env python3
# coding : utf-8


import os
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from functions import read_data, DataGenerator


BATCH_SIZE = 16
EPOCHS = 10


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--data_dir", required=True,
        help="Path to the images directory")
    ap.add_argument("-i", "--input", type=int, required=True, default=299,
        help="The input size")
    ap.add_argument("-t", "--tensorboard_path", required=True,
        help="Path to the model file")
    ap.add_argument("-c", "--checkpoint_dir", required=True,
        help="Path to the model weights checkpoints")
    ap.add_argument("-s", "--save_dir", required=True,
        help="Path to the model file")

    args = vars(ap.parse_args())
    size = args["input"]

    # data
    print("Loading data...")
    filenames, labels, num_classes = read_data(args["data_dir"])
    labels = tf.keras.utils.to_categorical(labels)

    train_set = DataGenerator(
        x_set=filenames,
        y_set=labels,
        batch_size=BATCH_SIZE,
        target_size=(size, size))
    steps_per_epoch = len(filenames) // BATCH_SIZE

    # ml
    print("Designing model...")
    base_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False, input_shape=(size, size, 3))
    base_model.trainable = False # freeze layers

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=1000, activation="relu"),
            tf.keras.layers.Dense(units=num_classes, activation="softmax")
        ])

    # checkpoints
    print("Loading weights...")
    if os.path.exists(args["checkpoint_dir"]):
        try:
            model.load_weights(args["checkpoint_dir"])
        except Exception as e:
            print("Oups!\nSomething turns wrong..\nMaybe Weights mismatch...", e)
    else:
        print("Weights not found")

    # cost function & optimization method
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"])

    # calllbacks
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=args["checkpoint_dir"],
        verbose=0,
        save_weights_only=True)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=args["tensorboard_path"],
        write_images=True)
    callbacks = [callback_checkpoint, callback_tensorboard]

    # training
    print("Start training...")
    history = model.fit(
        train_set,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks)

    # save
    print("Saving model...")
    now = time.localtime()
    model.save(os.path.join(
            args["save_dir"],
            f"model-{now.tm_mday}-{now.tm_mon}-{now.tm_year}.h5"))


if __name__ == "__main__":
    main()
