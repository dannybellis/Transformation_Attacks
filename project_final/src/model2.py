#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 02:24:45 2022

@author: daniel
"""
import numpy as np
import tensorflow as tf
from spatialTransformer import spatialTransformer
import time

def getModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train(model, X_train, y_train, sess, batch_size=50, training_steps=100000, attack=None):

    n_batches = int(len(X_train)/batch_size)
    X_train_batches = np.split(X_train, n_batches)
    y_train_batches = np.split(y_train, n_batches)
    
    start = time.time()
    for i in range(training_steps):
        idx = i%n_batches
        X_batch = X_train_batches[idx]
        y_batch = y_train_batches[idx]
        if attack != None and i>0:
            X_batch = attack.run(X_batch, y_batch, sess)
        metrics = model.train_on_batch(X_batch, y_batch, reset_metrics=False)
        end = time.time()
        if i%100==0 and i != 0:
            end = time.time()
            print("Step {}: Loss: {:.8f} Accuracy: {:.8f} Time: {:.4f}".format(i,metrics[0], metrics[1], end-start))
            start = time.time()
        
    
    return model