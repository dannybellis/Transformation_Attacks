#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 04:04:07 2022

@author: daniel
"""
import json

training_params = {
    "training steps" : 100000,
    "batch size" : 50,
    "adversary" : "none"
    }

spatial_params = {
    "XYRlimits" : [6,6,30],
    "step size" : 0.01,
    "epochs" : 200,
    "grid granularity" : [7, 7, 100],
    "random tries" : 1
    }

linf_params = {
    "epsilon" : 0.3,
    "step size" : 0.01,
    "epochs" : 40
    }

config = {
    "training parameters" : training_params,
    "spatial attack parameters" : spatial_params,
    "L infinity attack parameters" : linf_params
    }

with open('config.json', 'w') as config_file:
  json.dump(config, config_file, indent=4)

with open('config.json') as config_file:
    config = json.load(config_file)
  
training_params = config["training parameters"]
print(training_params)