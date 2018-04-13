# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:57:28 2018

@author: limchaos
"""

class GroupNormLayer():
    
    def __init__(
           self,
           prev_layer,
           act=tf.identity,
           G=4,
           gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
           beta_init=tf.zeros_initializer,
           eps=1e-5,
           name='GroupNormLayer',
    ):
        
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        logging.info("GroupNormLayer %s: act:%s" % (self.name, act.__name__))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]
        
        
        with tf.variable_scope(name) as vs:
            
            variables = []
            
            if beta_init:
                beta = tf.get_variable(
                       'beta',
                       shape=params_shape,
                       initializer=beta_init,
                       dtype=LayersConfig.tf.dtype,
                       trainable=True,
                )
            else:
                beta = None
            
            if gamma_init:
                gamma = tf.get_variable(
                       'gamma',
                       shape=params_shape,
                       initializer=gamma_init,
                       dtype=LayersConfig.tf.dtype,
                       trainable=True,
                )
                variables.append(gamma)
            else:
                gamma = None
                
            N, C, H, W = self.inputs.shape
            self.outputs = tf.reshape(self.inputs,[N, G, C // G, H, W])
            
            mean, var = tf.nn.moments(self.outputs,[2, 3, 4], keep_dim=True)
            self.outputs = (self.outputs - mean) / tf.sqrt(var + eps)
            self.outputs = tf.reshape(self.output,[N, C, H, W])
            self.outputs = x * gamma + beta
            self.outputs = act(self.outputs)
            
            self.all_layers.append(self.outputs)
            self.all_params.extend(variables)
            
            