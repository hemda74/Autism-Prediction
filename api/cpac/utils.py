#!/usr/bin/env python
import os
import re
import sys
import h5py
import time
import random
import string
import contextlib
import multiprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
from model import ae
from tensorflow.python.framework import ops


identifier = '(([a-zA-Z]_)?([a-zA-Z0-9_]*))'
replacement_field = '{' + identifier + '}'


def reset():
    ops.reset_default_graph()
    random.seed(19)
    np.random.seed(19)
    tf.random.set_random_seed(19)


def load_phenotypes(phenoPath):
    ph = pd.read_csv(phenoPath)
    ph = ph[ph['FILE_ID'] != 'no_filename']

    ph['DX_GROUP'] = ph['DX_GROUP'].apply(lambda v: int(v)-1)
    ph['SITE_ID'] = ph['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    ph['SEX'] = ph['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    ph['MEAN_FD'] = ph['func_mean_fd']
    ph['SUB_IN_SMP'] = ph['SUB_IN_SMP'].apply(lambda v: v == 1)
    ph["STRAT"] = ph[["SITE_ID", "DX_GROUP"]].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    ph.index = ph['FILE_ID']

    return ph[['FILE_ID', 'DX_GROUP', 'SEX', 'SITE_ID', 'MEAN_FD', 'SUB_IN_SMP', 'STRAT']]


def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename, fapl=propfaid)) as fid:
        f = h5py.File(fid, mode)
        return f

def load_fold(patients, exp, fold):

    derivative = exp.attrs["derivative"]

    XTrain = []
    YTrain = []
    for pid in exp[fold]["train"]:
        XTrain.append(np.array(patients[pid][derivative]))
        YTrain.append(patients[pid].attrs["y"])

    XValid = []
    Yvalid = []
    for pid in exp[fold]["valid"]:
        XValid.append(np.array(patients[pid][derivative]))
        Yvalid.append(patients[pid].attrs["y"])

    XTest = []
    YTest = []
    currPIDs = []
    for pid in exp[fold]["test"]:
        XTest.append(np.array(patients[pid][derivative]))
        YTest.append(patients[pid].attrs["y"])
        currPIDs.append(pid)
    currPIDs = np.array(currPIDs)
    currPIDs = currPIDs.tolist()
    currPIDs =[x.decode('utf-8') for x in currPIDs]

    return np.array(XTrain), YTrain, np.array(XValid), Yvalid, np.array(XTest), YTest, currPIDs


class SafeFormat(dict):

    def __missing__(self, key):
        return "{" + key + "}"

    def __getitem__(self, key):
        if key not in self:
            return self.__missing__(key)
        return dict.__getitem__(self, key)


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def format_config(s, *d):
    dd = merge_dicts(*d)
    return string.Formatter().vformat(s, [], SafeFormat(dd))


def elapsed_time(tstart):
    tnow = time.time()
    total = tnow - tstart
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def run_progress(callable_func, items, message=None, jobs=1):

    results = []

    print ('Starting pool of %d jobs' % jobs)

    current = 0
    total = len(items)

    if jobs == 1:
        results = []
        for item in items:
            results.append(callable_func(item))
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()

    else:
        pool = multiprocessing.Pool(processes=jobs)
        for item in items:
            pool.apply_async(callable_func, args=(item,), callback=results.append)

        while current < total:
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()
            time.sleep(0.5)

        pool.close()
        pool.join()

    return results


def root():
    return os.path.dirname(os.path.realpath(__file__))


def to_softmax(n_classes, classe):
    sm = [0.0] * n_classes
    sm[int(classe)] = 1.0
    return sm


def load_ae_encoder(input_size, code_size, model_path):
    model = ae(input_size, code_size)
    init = tf.global_variables_initializer()
    try:
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver(model["params"], write_version= tf.train.SaverDef.V2)
            if os.path.isfile(model_path):                
                saver.restore(sess, model_path)
            params = sess.run(model["params"])
            return {"W_enc": params["W_enc"], "b_enc": params["b_enc"]}
    finally:
        reset()


def sparsity_penalty(x, p, coeff):
    p_hat = tf.reduce_mean(tf.abs(x), 0)
    kl = p * tf.log(p / p_hat) + \
        (1 - p) * tf.log((1 - p) / (1 - p_hat))
    return coeff * tf.reduce_sum(kl)
