#! /usr/bin/env python3
# coding : utf-8


import os
import argparse
import json

import tqdm
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from functions import read_data, \
    load_model, load_image, extract_image_id


def encode_image(image):
    features = encoder(image)
    features = np.squeeze(features)

    return features

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--data_dir", required=True,
        help="Path to the images directory")
    ap.add_argument("-m", "--model_path", required=True,
        help="Path to the the model")
    ap.add_argument("-i", "--input", type=int, required=True, default=299,
        help="The input size")
    ap.add_argument("-o", "--output", required=True,
        help="Path to the output file")

    args = vars(ap.parse_args())
    size = args['input']

    # model
    print("Loading model...")
    subdir = args["model_path"]
    model_path = glob.glob(subdir+'*.h5')[-1]
    model = load_model(model_path)

    # data
    print("Reading data...")
    filenames, _, _ = read_data(args["data_dir"])
    n_files = len(filenames)

    # encoding
    print("Encoding images...")
    index_to_filename = {}
    filename_to_path = {}
    features = np.zeros((n_files, model.output.shape[1]))
    for i in tqdm.tqdm(range(n_files)):
        image_id = extract_image_id(filenames[i])
        index_to_filename[i] = image_id
        filename_to_path[image_id] = filenames[i]
        #print("->", image_id)
        image = load_image(filenames[i], (size, size))
        image = image.reshape((1,)+image.shape)

        features[i] = np.squeeze(model(image))

    # save transfer values
    np.save(args["output"], features)
    with open("index_to_filename.json", "w") as f:
        json.dump(index_to_filename, f, indent=4, ensure_ascii=False)
    with open("filename_to_path.json", "w") as f:
        json.dump(filename_to_path, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
