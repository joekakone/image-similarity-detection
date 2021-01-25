#! /usr/bin/env python3
# coding : utf-8


import os
import argparse
import time

from annoy import AnnoyIndex

from functions import load_transfer_values


def build_graph(features, size, n_trees, output):
    t = AnnoyIndex(size, 'angular')
    
    for i in range(len(features)):
        t.add_item(i, features[i])
    
    t.build(n_trees)
    
    now = time.localtime()
    t.save(os.path.join(
            output,
            f"graph-{now.tm_mday}-{now.tm_mon}-{now.tm_year}.ann"))
    
def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-i", "--input", required=True,
        help="Path to the input file")
    ap.add_argument("-s", "--size", type=int, required=True, default=2048,
        help="The input size")
    ap.add_argument("-t", "--tree", type=int, required=True,
        help="The number of trees in the graph")
    ap.add_argument("-o", "--output", required=True,
        help="Path to the output file")
    
    args = vars(ap.parse_args())

    # load tf values
    print("Loading tranfer values...")
    features = load_transfer_values(args['input'])

    # build annoy graph
    print("Building graph...")
    build_graph(features, args['size'], args['tree'], args['output'])


if __name__ == "__main__":
    main()
