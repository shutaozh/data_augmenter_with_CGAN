
import math
import json
import random
import pprint
import scipy.misc
import cv2
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import csv

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def load_data(data_files):
    """
    load data
    """
    data = []
    condition = []
    with open(data_files) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            results_data = list(map(int, row[0:520]))

            for i, item in enumerate(results_data):
                if results_data[i] > 100:
                    results_data[i] = 100
                elif results_data[i] < -100:
                    results_data[i] = -100
                else:
                    results_data[i] = results_data[i]

                results_data[i] = results_data[i] / 100

            data.append(results_data)

            results_condition = list(map(float, row[520:524]))

            condition.append(results_condition)
        min_item = min(min(row) for row in condition)
        max_item = max(max(row) for row in condition)
        for i, item in enumerate(condition):
            for j in range(4):
                condition[i][j] = (condition[i][j] - min_item) / (max_item - min_item)

    return np.concatenate([data, condition], axis=-1), min_item, max_item


def load_condition(data_files):
    """
    load condition
    """
    data = []
    condition = []
    with open(data_files) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            results_data = list(map(float, row[0:520]))

            data.append(results_data)

            results_condition = list(map(float, row[520:524]))

            condition.append(results_condition)
        min_item = min(min(row) for row in condition)
        max_item = max(max(row) for row in condition)
        for i, item in enumerate(condition):
            for j in range(4):
                condition[i][j] = (condition[i][j] - min_item) / (max_item - min_item)

    return np.concatenate([data, condition], axis=-1), min_item, max_item