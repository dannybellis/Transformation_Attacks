#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import path
import json
from sklearn.model_selection import train_test_split

from model2 import getModel, train
from attacks2 import linfPGD2, spatialGrid2, spatialPGD2, bayesianAttack, testTransform

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(77)
tf.set_random_seed(7)
# TARGET_MODEL_PATH = "../model/mnist_nn_natural2.h5"

def overall_evaluate(model, X_test, y_test):
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    return "Accuracy: {:.2f}%".format(acc*100)
    
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("python main.py [mode]")
        print("mode: {train, pgd}")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    with open('config.json') as config_file:
        config = json.load(config_file)
        
    training_params = config["training parameters"]
    model_path = "../model/"+ training_params["model path"]+".h5"
    spatial_params = config["spatial attack parameters"]
    linf_params = config["L infinity attack parameters"]
    
    (X_train, y_train),(X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    X_train  = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
# 
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=2000, stratify=y_test)
    with tf.Session() as sess:
        if mode == 'train':
            model = getModel()
            
            if training_params["adversary"] == "linf":
                attack = linfPGD2(model, linf_params["epsilon"], linf_params["step size"], linf_params["epochs"])
                
            elif training_params["adversary"] == "random":
                attack = spatialGrid2(model, limits=spatial_params["XYRlimits"], 
                                      granularity=spatial_params["grid granularity"],
                                      random_tries=1)
                
            elif training_params["adversary"] == "worst k":
                attack = spatialGrid2(model, limits=spatial_params["XYRlimits"], 
                                       granularity=spatial_params["grid granularity"],
                                       random_tries=spatial_params["random tries"])
            else:
                attack = None
            
            model = train(model, X_train, y_train, sess, 
                          batch_size=training_params["batch size"],
                          training_steps=training_params["training steps"],
                          attack = attack)
            
            model.save(model_path)
            
        if mode == "exp":
            model = tf.keras.models.load_model(model_path)
            
            print("Natural " + overall_evaluate(model, X_test, y_test))
            
            
            # attack_linf = linfPGD2(model, linf_params["epsilon"], linf_params["step size"], linf_params["epochs"])
            
            # linf_X = attack_linf.run(X_test, y_test, sess)
            
            # print("Linf " + overall_evaluate(model, linf_X, y_test))
            
            attack_random = spatialGrid2(model, limits=spatial_params["XYRlimits"], 
                                  granularity=spatial_params["grid granularity"],
                                  random_tries=1)
            
            random_X = attack_random.run(X_test, y_test, sess)
            
            print("Random " + overall_evaluate(model, random_X, y_test))
            
            attack_pgd = spatialPGD2(model, limits=spatial_params["XYRlimits"], 
                                  step_size=spatial_params["step size"],
                                  epochs=spatial_params["epochs"])
            
            pgd_X = attack_pgd.run(X_test, y_test, sess)
            
            print("PGD " + overall_evaluate(model, pgd_X, y_test))
            
            bayes = bayesianAttack(model = model, limits = spatial_params["XYRlimits"])
            
            bayes_X = bayes.run(X_test, y_test, sess)
            
            print("Bayes " + overall_evaluate(model, bayes_X, y_test))
            
            attack_grid = spatialGrid2(model, limits=spatial_params["XYRlimits"], 
                                  granularity=spatial_params["grid granularity"],
                                  random_tries=-1)
            
            grid_X = attack_grid.run(X_test, y_test, sess)
            
            print("Grid " + overall_evaluate(model, grid_X, y_test))
            
            
            
            
            
            
            
            # random_X = spatialGrid(model=target_model, X_seed=X_test, y_seed=y_test, sess=sess, 
            #                                       limits=spatial_params["XYRlimits"], 
            #                                       granularity=spatial_params["grid granularity"], 
            #                                       random_tries=spatial_params["random tries"])[0]
            
            # pgd_X = spatialPGD(model=target_model, X_seed=X_test, y_seed=y_test, sess=sess, 
            #                                       limits=spatial_params["XYRlimits"], 
            #                                       step_size=spatial_params["step size"], 
            #                                       epochs=spatial_params["epochs"])
            
            # grid_X = spatialGrid(model=target_model, X_seed=X_test, y_seed=y_test, sess=sess, 
            #                                       limits=spatial_params["XYRlimits"], 
            #                                       granularity=spatial_params["grid granularity"], 
            #                                       random_tries=-1)[0]
            
            # print("Natural " + overall_evaluate(model, X_test, y_test))
            # print("Random " + overall_evaluate(model, random_X, y_test))
            # print("PGD " + overall_evaluate(model, pgd_X, y_test))
            # print("Grid " + overall_evaluate(model, grid_X, y_test))
            
        # if mode == "train_adv":
        #     model = getModel()
        #     attack = linfPGD2(model, linf_params["epsilon"], linf_params["step size"], linf_params["epochs"])
        #     # target_model = tf.keras.models.load_model(TARGET_MODEL_PATH)
        #     model = train2(model, X_train, y_train, sess, batch_size=training_params["batch size"],
        #                    training_steps=training_params["training steps"],
        #                    attack = attack)
            
        # if mode == "test_adv1":
        #     model = tf.keras.models.load_model(TARGET_MODEL_PATH)
        #     attack = linfPGD2(model, linf_params["epsilon"], linf_params["step size"], linf_params["epochs"])
        #     X_adv = attack.run(X_train, y_train, sess)
        #     plt.imshow(X_adv[0])
        #     plt.show()
            
        # if mode == "test_adv2":
        #     model = tf.keras.models.load_model(TARGET_MODEL_PATH)
        #     attack = spatialGrid2(model, limits=spatial_params["XYRlimits"], 
        #                           granularity=spatial_params["grid granularity"],
        #                           random_tries=spatial_params["random tries"])
        #     X_adv = attack.run(X_train, y_train, sess)
        #     plt.imshow(X_adv[1])
        #     plt.show()
            
        # if mode == "test_adv3":
        #     model = tf.keras.models.load_model(TARGET_MODEL_PATH)
        #     attack = spatialPGD2(model, limits=spatial_params["XYRlimits"], 
        #                           step_size=spatial_params["step size"],
        #                           epochs=spatial_params["epochs"])
        #     X_adv = attack.run(X_train, y_train, sess)
        #     plt.imshow(X_adv[1])
        #     plt.show()
            
            
            