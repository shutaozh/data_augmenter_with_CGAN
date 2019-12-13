import csv
import numpy as np
import sys
import tarfile
import math
import os
import scipy.stats
from functools import reduce
import random


# load data
def load_data(data_files):
    data = []
    condition = []
    with open(data_files) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            results_data = list(map(float, row[0:520]))

            for i, item in enumerate(results_data):
                if results_data[i] > 100:
                    results_data[i] = 100
                elif results_data[i] < -100:
                    results_data[i] = -100
                else:
                    results_data[i] = results_data[i]

            data.append(results_data)

            results_condition = list(map(float, row[520:523]))

            condition.append(results_condition)

    return np.array(data), np.array(condition)


# calculate mean and variance
def mean_variance(datas, conditions):
    condition = np.unique(conditions, axis=0)
    means = []
    variances = []
    for i in range(len(condition)):
        index = np.where((conditions == condition[i]).all(1))[0]
        data = []
        var = []
        for j in index:
            data.append(datas[j])
        x = np.array(data)
        y = np.array(data)
        x[x == 100] = 0
        y[y == 100] = 0
        y[y < 0] = 1
        s = x.sum(axis=0)  # sum of negative
        num = y.sum(axis=0)  # num of negative
        mean = s / (num + 0.0001)

        for m in range(M):
            num1 = 0
            s1 = 0
            for n in range(len(x)):
                if x[n, m] < 0:
                    s1 += np.square(x[n, m] - mean[m])
                    num1 += 1
            var.append(s1 / (num1 + 0.0001))
        variance = np.array(var)

        means.append(mean)
        variances.append(variance)
    means = np.array(means)
    variances = np.array(variances)

    return means, variances, condition


# Gaussian function
def gaussian(x, mu, delta):
    p = []
    for i in range(len(x)):
        if x[i] < 0:
            if delta[i] == 0:
                p.append(1.0)
            else:
                p.append(scipy.stats.norm(mu[i], delta[i]).pdf(x[i]))

    return p


# HP Score, both real and fake data are negative
def hp(real, fake):
    num = 0
    for i in range(NB):
        nh = 0
        m = 0
        for j in range(M):
            if real[i][j] < 0 and fake[i][j] < 0:
                nh += 1
            if real[i][j] < 0:
                m += 1
        num = num + (nh + 0.0001) / (m + 0.0001)
    return num / NB


# RFP Score, the fake data within range (mean - variance, mean + variance)
def rf(fake_data, fake_condition):
    a = 0
    for i in range(len(condition)):
        index = np.where((fake_condition == condition[i]).all(1))[0]
        for j in index:
            if (fake_condition[j] == condition[i]).all():
                num = 0
                n = 0
                for k in range(M):
                    if fake_data[j][k] < 0:
                        num += 1
                        if (mean[i][k] - variance[i][k]) <= fake_data[j][k] <= (mean[i][k] + variance[i][k]):
                            n += 1
                a += n / (num + 0.0001)
    return a / NB


# DP Score, noise number rate
def dp(real, fake):
    num = 0
    for i in range(NB):
        nh = 0
        for j in range(M):
            if real[i][j] < -90 and fake[i][j] > 0:
                nh += 1
            if fake[i][j] < -90 and real[i][j] > 0:
                nh += 1
        num = num + nh
    return num / NB


# inception score
def ris(fake_data, fake_condition):
    pyx = []
    for i in range(len(fake_data)):
        x = []
        for j in range(len(condition)):
            index = np.where((fake_condition == condition[j]).all(1))[0]
            pyi = len(index) / len(fake_condition)
            p = gaussian(fake_data[i], mean[j], np.sqrt(variance[j]))
            if p:
                pxy = reduce(lambda a, b: a * b, p)
                x.append(pxy * pyi)
            else:
                x.append(1.0 * pyi)
        pyx.append(x / (np.sum(x)))

    ppp = np.array(pyx)
    py = np.mean(ppp, axis=0)
    KL = []
    for i in range(len(ppp)):
        KL.append(scipy.stats.entropy(ppp[i], py))
    k = np.array(KL)
    zz = np.mean(k)

    return np.exp(zz)


real_data, real_condition = load_data('data/ValidationData.csv')
fake_data, fake_condition = load_data('data/fake_data.csv')

index = random.sample(range(len(real_data)), 1000)
real_data = real_data[index]
real_condition = real_condition[index]
fake_data = fake_data[index]
fake_condition = fake_condition[index]

NB = len(real_data)
M = 520
mean, variance, condition = mean_variance(real_data, real_condition)

HP = hp(real_data, fake_data)
print('HP: '+str(HP))

RF = rf(fake_data, fake_condition)
print('RF: '+str(RF))

DP = dp(real_data, fake_data)
print('DP: '+str(DP))


RIS = ris(fake_data, fake_condition)
print('RIS: '+str(RIS))
