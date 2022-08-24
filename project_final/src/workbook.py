#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel Ellis
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import path
import json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(0)
tf.set_random_seed(0)
TARGET_MODEL_PATH = "../model/mnist_nn.h5"

def overall_evaluate(model, X_test, y_test):
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print("Accuracy: {:.2f}".format(acc))

def train(X_train, y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=128)
    return model

def getPixelValue(img, x, y):
    bs = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    
    batch_idx = tf.range(0, bs)
    batch_idx = tf.reshape(batch_idx, (bs, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))
    
    indices = tf.stack([b,y,x],3)
    
    return tf.gather_nd(img, indices)

def gridGenerator(height, width, theta):
    bs = tf.shape(theta)[0]
    
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    
    x_t, y_t = tf.meshgrid(x, y)
    
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    
    ones = tf.ones_like(x_t_flat)
    
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([bs, 1, 1]))
    sampling_grid = tf.cast(sampling_grid, 'float32')
    
    theta = tf.cast(theta, 'float32')
    
    batch_grids = tf.matmul(theta, sampling_grid)
    batch_grids = tf.reshape(batch_grids, [bs, 2, height, width])
    
    return batch_grids

def bilinearSampler(img, x, y):
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    
    max_y = tf.cast(H-1, 'int32')
    max_x = tf.cast(W-1, 'int32')
    
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    x = 0.5*((x+1.0)*tf.cast(max_x-1, 'float32'))
    y = 0.5*((y+1.0)*tf.cast(max_y-1, 'float32'))
    
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0+1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0+1
    
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)
    
    Ia = getPixelValue(img, x0, y0)
    Ib = getPixelValue(img, x0, y1)
    Ic = getPixelValue(img, x1, y0)
    Id = getPixelValue(img, x1, y1)
    
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    
    wa = (x1-x)*(y1-y)
    wb = (x1-x)*(y-y0)
    wc = (x-x0)*(y1-y)
    wd = (x-x0)*(y-y0)
    
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)
    
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    
    return out

def spatialTransformer(X, rot, tx, ty):
    B = tf.shape(X)[0]
    H = tf.shape(X)[1]
    W = tf.shape(X)[2]
    
    theta1 = tf.stack([tf.cos(rot), -1*tf.sin(rot), tx], axis=-1)
    theta1 = tf.reshape(theta1, [B, 1, 3])
    
    theta2 = tf.stack([tf.sin(rot), tf.cos(rot), ty], axis=-1)
    theta2 = tf.reshape(theta2, [B, 1, 3])

    theta = tf.stack([theta1, theta2], axis=2)
    theta = tf.reshape(theta, [B, 2, 3])
    
    batch_grids = gridGenerator(H, W, theta)
    
    xgrid = batch_grids[:,0,:,:]
    ygrid = batch_grids[:,1,:,:]
    
    fmap = bilinearSampler(X, xgrid, ygrid)
    
    return fmap
    
def spatialPGD(model, X_seed, y_seed, sess, rot_lim=30.0, tr_lim=3.0, step_size=0.01, epochs=200):
    bn = X_seed.shape[0]
    h = X_seed.shape[1]
    w = X_seed.shape[2]
    
    rot_lim = rot_lim*np.pi/180.0
    tr_lim_x = 2.0*tr_lim/float(w)
    tr_lim_y = 2.0*tr_lim/float(h)
    
    X_input = tf.placeholder('float32', [None, h, w, 1])
    y_true = tf.placeholder('uint8', [None])
    
    rot = tf.placeholder('float32', [None])
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
    
    r = np.random.uniform(low=-rot_lim, high=rot_lim, size=bn)
    t_x = np.random.uniform(low=-tr_lim_x, high=tr_lim_x, size=bn)
    t_y = np.random.uniform(low=-tr_lim_y, high=tr_lim_y, size=bn)
    
    X_adv = None
    for i in range(epochs):
        
        feedDict= {X_input : X_seed,
                   y_true : y_seed,
                   rot : r,
                   tx : t_x,
                   ty : t_y}
        
        r, t_x, t_y, X_adv = sess.run([rot_update, tx_update, ty_update, fmap], feed_dict=feedDict)
        acc = model.evaluate(X_adv, y_seed, verbose=0)[1]
        print(i, " Accuracy: {:.3f}".format(acc))
        
    return X_adv
        
def linfPGD(model, X_seed, y_seed, sess, epsilon=0.3, step_size=0.01, epochs=40):
    output = model.output
    y_true = tf.placeholder("uint8", [None])
    loss = tf.keras.backend.sparse_categorical_crossentropy(y_true, output)
    grad = tf.gradients(loss, model.input)
    adv_update_noclip = target_model.input+step_size*tf.math.sign(grad)
    adv_update_linf = tf.clip_by_value(adv_update_noclip, X_seed-epsilon, X_seed+epsilon)
    adv_update = tf.clip_by_value(adv_update_linf, 0, 1)
    
    X_adv = X_seed
    for i in range(epochs):
        X_adv = sess.run(adv_update, feed_dict={y_true: y_seed, target_model.input: X_adv})[0]
        acc = target_model.evaluate(X_adv, y_seed, verbose=0)[1]
        print("Accuracy: {:.2f}".format(acc))
        
    return X_adv

def spatialRandom(model, X_seed, y_seed, sess, rot_lim=30.0, tr_lim=3.0):
    X_adv = spatialPGD(model, X_seed, y_seed, sess, rot_lim=rot_lim, tr_lim=tr_lim, step_size=0.01, epochs=1)
    return X_adv

def spatialGrid(model, X_seed, y_seed, sess, rot_lim=30.0, tr_lim=3.0):
    pass
    
    
    
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("python main.py [mode]")
        print("mode: {train, pgd}")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    (X_train, y_train),(X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    X_train  = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    with tf.Session() as sess:
        if mode == 'train':
            model = train(X_train, y_train)
            
            overall_evaluate(model, X_test, y_test)
            
            model.save(TARGET_MODEL_PATH)
            
        elif mode == 'spatial_pgd':
            target_model = tf.keras.models.load_model(TARGET_MODEL_PATH)
            
            X_adv = spatialPGD(target_model, X_test, y_test, sess)
            
            preds = target_model.predict(X_adv).argmax(axis=1)
            evasives = np.where(y_test != preds)[0]
            for i in evasives[:10]:
                plt.imshow(X_adv[i], cmap = 'gray')
                plt.savefig("../result/spatial_attack/Figure{}-{}-{}.png".format(i, y_test[i], preds[i]))
                
        elif mode == 'linf_pgd':
            target_model = tf.keras.models.load_model(TARGET_MODEL_PATH)
            
            X_adv = linfPGD(target_model, X_test, y_test, sess)
            
            preds = target_model.predict(X_adv).argmax(axis=1)
            evasives = np.where(y_test != preds)[0]
            for i in evasives[:10]:
                plt.imshow(X_adv[i], cmap = 'gray')
                plt.savefig("../result/linf_attack/Figure{}-{}-{}.png".format(i, y_test[i], preds[i]))
            
        
            
