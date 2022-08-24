#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from spatialTransformer import spatialTransformer
from itertools import product, repeat
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# np.random.seed(0)
# tf.set_random_seed(0)

def spatialPGD(model, X_seed, y_seed, sess, limits=[6,6,30], step_size=0.01, epochs=200):
    
    bn = X_seed.shape[0]
    h = X_seed.shape[1]
    w = X_seed.shape[2]
    
    rot_lim = np.radians(limits[2])
    tr_lim_x = limits[0]/float(w)
    tr_lim_y = limits[1]/float(h)
    
    X_input = tf.placeholder('float32', [None, h, w, 1])
    y_true = tf.placeholder('uint8', [None])
    
    rot = tf.placeholder(tf.float32, [None])
    tx = tf.placeholder('float32', [None])
    ty = tf.placeholder('float32', [None])
    
    fmap = spatialTransformer(X_input, rot, tx, ty)
    
    pred = model(fmap, training=False)
    
    loss = tf.keras.backend.sparse_categorical_crossentropy(y_true, pred)
    
    grad_rot, grad_tx, grad_ty = tf.gradients(loss, [rot, tx, ty])
    
    rot_update_noclip = rot + step_size*grad_rot
    rot_update = tf.clip_by_value(rot_update_noclip, -rot_lim, rot_lim)
    
    tx_update_noclip = tx + step_size*grad_tx
    tx_update = tf.clip_by_value(tx_update_noclip, -tr_lim_x, tr_lim_x)
    
    ty_update_noclip = ty + step_size*grad_ty
    ty_update = tf.clip_by_value(ty_update_noclip, -tr_lim_y, tr_lim_y)
    
    # r = np.random.uniform(low=-rot_lim, high=rot_lim, size=bn).astype(np.float32)
    # t_x = np.random.uniform(low=-tr_lim_x, high=tr_lim_x, size=bn).astype(np.float32)
    # t_y = np.random.uniform(low=-tr_lim_y, high=tr_lim_y, size=bn).astype(np.float32)
    # r_ = np.copy(r)
    # t_x_ = t_x
    # t_y_ = t_y
    r = np.zeros((bn))
    t_x = np.zeros((bn))
    t_y = np.zeros((bn))
    
    r_ = np.zeros((bn))
    t_x_ = np.zeros((bn))
    t_y_ = np.zeros((bn))
    
    X_adv = None
    for i in range(epochs):
        
        feedDict2= {X_input : X_seed,
                   y_true : y_seed,
                   rot : r_,
                   tx : t_x_,
                   ty : t_y_}
        
        g_r, g_tx, g_ty= sess.run([grad_rot, grad_tx, grad_ty], feed_dict=feedDict2)
        
        # g_r, g_tx, g_ty, X_adv = sess.run([grad_rot, grad_tx, grad_ty, fmap], feed_dict=feedDict)

        r_ = r_ + step_size*g_r
        # r_ = np.clip(r_, -rot_lim, rot_lim)
        print(r_)
        t_x_ = t_x_ + step_size*g_tx
        # t_x_ = np.clip(t_x_, -tr_lim_x, tr_lim_x)
        
        t_y_ = t_y_ + step_size*g_ty
        # t_y_ = np.clip(t_y_, -tr_lim_y, tr_lim_y)
        
        feedDict1= {X_input : X_seed,
                   y_true : y_seed,
                   rot : r,
                   tx : t_x,
                   ty : t_y}
        
        r, t_x, t_y = sess.run([rot_update, tx_update, ty_update], feed_dict=feedDict1)
        # print(abs(r_-r))
        # print(r_)
        # print(all(np.equal(r, r_)),all(np.equal(t_x, t_x_)),all(np.equal(t_y, t_y_)))
        
    
        # acc = model.evaluate(X_adv, y_seed, verbose=0)[1]
        
        # print("PGD epoch {} Accuracy: {:.2f}".format(i, acc*100))
       
    return X_adv

def spatialGrid(model, X_seed, y_seed, sess, limits=[3,3,30], granularity=[7,7,100], random_tries=-1):
    bn = X_seed.shape[0]
    h = X_seed.shape[1]
    w = X_seed.shape[2]

    X_input = tf.placeholder('float32', [None, h, w, 1])
    y_true = tf.placeholder('uint8', [None])
    
    rot = tf.placeholder('float32', [None])
    tx = tf.placeholder('float32', [None])
    ty = tf.placeholder('float32', [None])
    
    fmap = spatialTransformer(X_input, rot, tx, ty)
    
    pred = model(fmap, training=False)
    
    xent = tf.keras.backend.sparse_categorical_crossentropy(y_true, pred)
    
    if random_tries>0:
        grid = [limits for i in range(random_tries)]
    else:   
        grid = product(*list(np.linspace(-l, l, num=g) for l,g in zip(limits, granularity)))
    
    max_xent = np.zeros(bn)
    all_correct = np.ones(bn).astype(bool)
    worst_tx = np.zeros(bn)
    worst_ty = np.zeros(bn)
    worst_r = np.zeros(bn)

    for t_x_, t_y_, r_ in grid:
        if random_tries>0:
            r = np.radians(np.random.uniform(-r_, r_, bn))
            t_x = 2*np.random.uniform(-t_x_, t_x_, bn)/float(w)
            t_y = 2*np.random.uniform(-t_y_, t_y_, bn)/float(h)
            
        else:
            r = np.radians(np.repeat(r_, bn))
            t_x = 2*np.repeat(t_x_, bn)/float(w)
            t_y = 2*np.repeat(t_y_, bn)/float(h)
            print("Grid Transformation {:.2f}, {:.2f}, {:.2f}".format(t_x_, t_y_, r_))
        
        
        feedDict = {X_input : X_seed,
                   y_true : y_seed,
                   rot : r,
                   tx : t_x,
                   ty : t_y}
        
        curr_xent, preds = sess.run([xent, pred], feed_dict=feedDict)
        
        curr_correct = (preds.argmax(axis=1) == y_seed)
        
        idx = (curr_xent >= max_xent) & (curr_correct == all_correct)
        idx = idx | (curr_correct < all_correct)
        max_xent = np.maximum(max_xent, curr_xent)
        all_correct = curr_correct & all_correct
        
        worst_tx = np.where(idx, t_x, worst_tx)
        worst_ty = np.where(idx, t_y, worst_ty)
        worst_r = np.where(idx, r, worst_r)
    
    feedDict = {X_input : X_seed,
               y_true : y_seed,
               rot : worst_r,
               tx : worst_tx,
               ty : worst_ty}

    X_adv = sess.run(fmap, feed_dict=feedDict)
    
    return X_adv, worst_tx, worst_ty, worst_r

def linfPGD(model, X_seed, y_seed, sess, epsilon=0.3, step_size=0.01, epochs=40):
    
    output = model.output
    y_true = tf.placeholder("uint8", [None])
    loss = tf.keras.backend.sparse_categorical_crossentropy(y_true, output)
    grad = tf.gradients(loss, model.input)
    adv_update_noclip = model.input+step_size*tf.math.sign(grad)
    adv_update_linf = tf.clip_by_value(adv_update_noclip, X_seed-epsilon, X_seed+epsilon)
    adv_update = tf.clip_by_value(adv_update_linf, 0, 1)

    X_adv = X_seed
    for i in range(epochs):
        X_adv = sess.run(adv_update, feed_dict={y_true: y_seed, model.input: X_adv})[0]
    
    return X_adv