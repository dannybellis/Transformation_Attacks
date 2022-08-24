#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 23:00:18 2022

@author: daniel

"""
import numpy as np
import tensorflow as tf
from spatialTransformer import spatialTransformer
from itertools import product, repeat
import matplotlib.pyplot as plt
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm
from scipy.special import erf
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


np.random.seed(77)
tf.set_random_seed(7)

class linfPGD2():
    def __init__(self, model, epsilon, step_size, epochs):
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.epochs = epochs
        
        self.output = self.model.output
        self.y_true = tf.placeholder("uint8", [None])
        self.loss = tf.keras.backend.sparse_categorical_crossentropy(self.y_true, self.output)
        self.grad = tf.gradients(self.loss, self.model.input)
        
    def run(self, X, y, sess):
        X_adv = np.copy(X)
        for i in range(self.epochs):
            grad = sess.run(self.grad, feed_dict = {self.y_true: y, self.model.input: X_adv})[0]
            X_adv = X_adv+self.step_size*np.sign(grad)
            X_adv = np.clip(X_adv, X-self.epsilon, X+self.epsilon)
            X_adv = np.clip(X_adv, 0, 1)
        return X_adv
        
class spatialGrid2():
    def __init__(self, model, limits, granularity, random_tries):
        self.model = model
        self.limits = limits
        self.granularity = granularity
        self.random_tries = random_tries
        
        self.X_input = tf.placeholder('float32', [None, None, None, 1])
        self.y_true = tf.placeholder('uint8', [None])
        
        self.rot = tf.placeholder('float32', [None])
        self.tx = tf.placeholder('float32', [None])
        self.ty = tf.placeholder('float32', [None])
        
        self.fmap = spatialTransformer(self.X_input, self.rot, self.tx, self.ty)
        
        self.pred = self.model(self.fmap, training=False)
        
        self.xent = tf.keras.backend.sparse_categorical_crossentropy(self.y_true, self.pred)
    
    def run(self, X, y, sess):
        
        bn = X.shape[0]
        h = X.shape[1]
        w = X.shape[2]
        
        if self.random_tries>0:
            grid = [self.limits for i in range(self.random_tries)]
        else:   
            grid = product(*list(np.linspace(-l, l, num=g) for l,g in zip(self.limits, self.granularity)))
        
        max_xent = np.zeros(bn)
        all_correct = np.ones(bn).astype(bool)
        worst_tx = np.zeros(bn)
        worst_ty = np.zeros(bn)
        worst_r = np.zeros(bn)
                
        for t_x_, t_y_, r_ in grid:
            
            if self.random_tries>0:
                r = np.radians(np.random.uniform(-r_, r_, bn))
                t_x = 2*np.random.uniform(-t_x_, t_x_, bn)/float(w)
                t_y = 2*np.random.uniform(-t_y_, t_y_, bn)/float(h)
                
            else:
                r = np.radians(np.repeat(r_, bn))
                t_x = 2*np.repeat(t_x_, bn)/float(w)
                t_y = 2*np.repeat(t_y_, bn)/float(h)
                
            feedDict = {self.X_input : X,
                       self.y_true : y,
                       self.rot : r,
                       self.tx : t_x,
                       self.ty : t_y}
            
            curr_xent, preds = sess.run([self.xent, self.pred], feed_dict=feedDict)
            
            curr_correct = (preds.argmax(axis=1) == y)
            
            idx = (curr_xent > max_xent) & (curr_correct == all_correct)
            idx = idx | (curr_correct < all_correct)
            max_xent = np.maximum(max_xent, curr_xent)
            all_correct = curr_correct & all_correct
            worst_tx = np.where(idx, t_x, worst_tx)
            worst_ty = np.where(idx, t_y, worst_ty)
            worst_r = np.where(idx, r, worst_r)
            
        feedDict = {self.X_input : X,
                   self.y_true : y,
                   self.rot : worst_r,
                   self.tx : worst_tx,
                   self.ty : worst_ty}
        
        X_adv = sess.run(self.fmap, feed_dict=feedDict)
        
        return X_adv
    
class spatialPGD2():
    def __init__(self, model, limits, step_size, epochs):
        self.model = model
        self.limits = limits
        self.step_size = step_size
        self.epochs = epochs
        
        self.X_input = tf.placeholder('float32', [None, None, None, 1])
        self.y_true = tf.placeholder('uint8', [None])
        
        self.rot = tf.placeholder('float32', [None])
        self.tx = tf.placeholder('float32', [None])
        self.ty = tf.placeholder('float32', [None])
        
        self.fmap = spatialTransformer(self.X_input, self.rot, self.tx, self.ty)
        
        self.pred = self.model(self.fmap, training=False)
        
        self.loss = tf.keras.backend.sparse_categorical_crossentropy(self.y_true, self.pred)
        
        self.grad_rot, self.grad_tx, self.grad_ty = tf.gradients(self.loss, [self.rot, self.tx, self.ty])
    
        
    def run(self, X, y, sess):
        bn = X.shape[0]
        h = X.shape[1]
        w = X.shape[2]
        
        tx_lim = 2*self.limits[0]/float(w)
        ty_lim = 2*self.limits[1]/float(h)
        rot_lim = np.radians(self.limits[2])
        
        rot_ = np.random.uniform(low=-rot_lim, high=rot_lim, size=bn)
        tx_ = np.random.uniform(low=-tx_lim, high=tx_lim, size=bn)
        ty_ = np.random.uniform(low=-ty_lim, high=ty_lim, size=bn)
        
        X_adv = None
        
        for i in range(self.epochs):
            feedDict= {self.X_input : X,
                       self.y_true : y,
                       self.rot : rot_,
                       self.tx : tx_,
                       self.ty : ty_}
            
            grad_rot_, grad_tx_, grad_ty_, X_adv = sess.run([self.grad_rot, self.grad_tx, self.grad_ty, self.fmap], feed_dict=feedDict)
            
            tx_ = tx_ + self.step_size*grad_tx_
            tx_ = np.clip(tx_,-tx_lim, tx_lim)
            
            ty_= ty_ + self.step_size*grad_ty_
            ty_ = np.clip(ty_,-ty_lim, ty_lim)
            
            rot_ = rot_ + self.step_size*grad_rot_
            rot_ = np.clip(rot_,-rot_lim, rot_lim)

        
            # acc = self.model.evaluate(X_adv, y, verbose=0)[1]
            # print("Accuracy at epoch {}: {:.2f}".format(i,acc*100))
        
        return X_adv
    
class bayesianAttack():
    def __init__(self, model, limits, epochs=10, epsilon=0.1, priors=1, n_restarts=25):
        self.model = model
        self.limits = np.asarray(limits)
        self.limits[2] = np.radians(self.limits[2])
        self.limits[:2] = self.limits[:2]*2/28
        self.opt_limits = [(-i, i) for i in self.limits]
        self.epochs = epochs
        self.n_restarts = n_restarts 
        self.priors = priors
        
        self.X_input = tf.placeholder('float32', [None, None, None, 1])
        self.y_true = tf.placeholder('uint8', [None])
        
        self.rot = tf.placeholder('float32', [None])
        self.tx = tf.placeholder('float32', [None])
        self.ty = tf.placeholder('float32', [None])
        
        self.fmap = spatialTransformer(self.X_input, self.rot, self.tx, self.ty)
        
        self.pred = self.model(self.fmap, training=False)
        
        self.loss = tf.keras.backend.sparse_categorical_crossentropy(self.y_true, self.pred)
    
    def ei(self, X, y, gp):
        mu, sigma = gp.predict(X, return_std=True)
        y_max = np.max(y)
        with np.errstate(divide='ignore'):
            z = (mu-y_max)/sigma
            ei = sigma*norm.pdf(z)+(mu-y_max)*norm.cdf(z)
            return ei
    
    def nextSample(self, gp, y):
        x_opt = None
        af_opt = np.Inf
        
        af = lambda x, y, gp : -1*self.ei(x.reshape(1,-1), y, gp)
        
        grid = np.random.uniform(-self.limits, self.limits, size=(self.n_restarts, 3))
        for x0 in grid:
            ei_res = minimize(af, x0=x0, bounds=self.opt_limits, args=(y, gp))
            if ei_res.fun < af_opt:
                af_opt = ei_res.fun
                x_opt = ei_res.x
        
        return x_opt
    
    def getSample(self, X, y, t, sess):
        
        t = t.reshape(-1,1)
        
        feedDict = {self.X_input : X,
                   self.y_true : y,
                   self.rot : t[2],
                   self.tx : t[0],
                   self.ty : t[1]}
        
        loss = sess.run(self.loss, feed_dict=feedDict)[0]
        
        return loss

    def bayesianSubroutine(self, X, y, sess):
        
        x_priors = np.random.uniform(-self.limits, self.limits, size=(self.priors, 3))
        y_priors = np.array([])
        
        for prior in x_priors:
            y_prior = self.getSample(X, y, prior, sess)
            y_priors = np.append(y_priors, y_prior)
            
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel)
        
        for i in range(self.epochs):
            gp.fit(x_priors, y_priors)
            next_x = self.nextSample(gp=gp, y=y_priors)
            next_y = self.getSample(X, y, next_x, sess)
            x_priors = np.append(x_priors, next_x.reshape(1,-1), axis=0)    
            y_priors = np.append(y_priors, next_y)
        
        best_idx = np.argmax(y_priors)
        best_x = x_priors[best_idx]
        best_y = y_priors[best_idx]
        
        return best_x, best_y
        
    
    def run(self, X, y, sess):
        y_reshape = y.reshape(-1,1)
        transformations = []
        
        for x,y_ in zip(X,y_reshape):
            x = x.reshape(1,x.shape[0], x.shape[1], x.shape[2])
            t, l = self.bayesianSubroutine(x, y_, sess)
            transformations.append(t)
            
        transformations = np.asarray(transformations)
        transformations = transformations.T
        
        feedDict = {self.X_input : X,
                    self.y_true : y,
                    self.rot : transformations[2],
                    self.tx : transformations[0],
                    self.ty : transformations[1]}
        
        
        X_adv = sess.run(self.fmap, feed_dict = feedDict)
        
        return X_adv
    
class testTransform():
    def __init__(self, model):
        
        self.model = model
        
        self.X_input = tf.placeholder('float32', [None, None, None, 1])
        self.y_true = tf.placeholder('uint8', [None])
        
        self.rot = tf.placeholder('float32', [None])
        self.tx = tf.placeholder('float32', [None])
        self.ty = tf.placeholder('float32', [None])
        
        self.fmap = spatialTransformer(self.X_input, self.rot, self.tx, self.ty)
        
        self.pred = self.model(self.fmap, training=False)
        
        self.xent = tf.keras.backend.sparse_categorical_crossentropy(self.y_true, self.pred)
        
    def run(self, X, y, t, sess):
        
        feedDict = {self.X_input : X,
                    self.y_true : y,
                    self.rot : t[2],
                    self.tx : t[0],
                    self.ty : t[1]}
        
        loss = sess.run(self.xent, feed_dict=feedDict)
        
        return loss

