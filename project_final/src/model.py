#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from spatialTransformer import spatialTransformer
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train(X_train, y_train, batch_size=50, training_steps=100000, attack=None):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    n_batches = int(len(X_train)/batch_size)
    X_train_batches = np.split(X_train, n_batches)
    y_train_batches = np.split(y_train, n_batches)
    
    model.train_on_batch(X_train_batches[0], y_train_batches[0], reset_metrics=False)
        
    for i in range(training_steps):
        start = time.time()
        idx = i%n_batches
        X_batch = X_train_batches[idx]
        y_batch = y_train_batches[idx]
        if attack != None:
            X_batch = attack(model, X_batch, y_batch)
            
        metrics = model.train_on_batch(X_batch, y_batch, reset_metrics=False)
        end = time.time()
        # if i%10==0:
        print("Step {}: Loss: {:.8f} Accuracy: {:.8f}".format(i,metrics[0], metrics[1]))
        print(end-start)
    
    return model

def getModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train2(model, X_train, y_train, sess, batch_size=50, training_steps=100000, attack=None):

    n_batches = int(len(X_train)/batch_size)
    X_train_batches = np.split(X_train, n_batches)
    y_train_batches = np.split(y_train, n_batches)
    
    model.train_on_batch(X_train_batches[0], y_train_batches[0], reset_metrics=False)
        
    for i in range(training_steps):
        start = time.time()
        idx = i%n_batches
        X_batch = X_train_batches[idx]
        y_batch = y_train_batches[idx]
        if attack != None:
            X_batch = attack.run(X_batch, y_batch, sess)
            
        metrics = model.train_on_batch(X_batch, y_batch, reset_metrics=False)
        end = time.time()
        # if i%10==0:
        print("Step {}: Loss: {:.8f} Accuracy: {:.8f}".format(i,metrics[0], metrics[1]))
        print(end-start)
    
    return model