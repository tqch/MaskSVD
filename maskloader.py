"""
Select nearest neighbors for target user based on the percentage of
similar attitudes towards books in the intersection (books rated by both)
generate specific subsets for training and test for recommendation
"""

import os
import numpy as np

from train_test_splitter import *

def loader(file1,file2):
    # load data from npz file
    with np.load(file1) as data:
        ratings_train = data["ratings_train"]
    with np.load(file2) as data:
        ratings_test = data["ratings_test"]
    # recover the mask information
    mask = np.isnan(ratings_train)
    unmask = 1-np.isnan(ratings_test)
    ratings_train = np.nan_to_num(ratings_train)
    ratings_test = np.nan_to_num(ratings_test)
    return ratings_train,ratings_test,mask,unmask

def similarity(tid,rid,like,mask):
    """
    return a function computing the similarity given two ids
    inputs:
        tid: target user id (row index)
        rid: referred user id (row index)
        mask: characteristic matrix of the unobserved
    """
    # assume similarity to be 2 when rid = tid
    # so that tid itself will always be on the last element of nn indices
    if rid==tid:
        return np.inf
    else:
        tobs = 1-mask[tid]
        robs = 1-mask[rid]
        overlap = np.nonzero(tobs*robs)[0]
        if len(overlap) == 0:
            return 0
        else:
            return np.sum(np.equal(like[tid],like[rid])[overlap])/len(overlap)

def masknn(like,mask,tid,k=100):
    """
    find k nearest neighbors of target user
    return: row indices
    """
    n = like.shape[0]
    smlrt = np.zeros(n)
    for rid in range(n):
        smlrt[rid] = similarity(tid,rid,like,mask)
    return np.argsort(smlrt)[-k::1]

def maskloader(tid=None,cache=None,mode="none"):
    # use cache to accelerate loading
    if cache == None:
        file1 = "processed\\ratings_train.npz"
        file2 = "processed\\ratings_test.npz"
        x_train, x_test, mask, unmask = loader(file1,file2)
    else:
        x_train, x_test, mask, unmask = cache
    baseline = np.sum(x_train,axis=1)/np.sum(1-mask,axis=1)
    # determine whether like or not by the baseline (average rating)
    x = x_train-baseline[:,np.newaxis]
    like = (x >= 0).astype(np.int32)
    y = ((x_test-baseline[:,np.newaxis])>=0).astype(np.int32)
    # turn standardized rating matrix to likeness matrix (0/1)
    if mode == "binary":
        x = like
    if tid == None:
        return x, y, mask, unmask
    else:
        nn = masknn(like,mask,tid)
        return x[nn], y[nn], mask[nn], unmask[nn]