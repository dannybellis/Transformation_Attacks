#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

