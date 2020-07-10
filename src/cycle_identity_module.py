# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:52:02 2018

@author: yeohyeongyu
"""
import tensorflow as tf
from tensorflow.keras import layers


def discriminator(image_shape, options, name='discriminator'):
    def first_layer(input_, out_channels, ks=3, s=1):
        return lrelu(conv2d(input_, out_channels, ks=ks, s=s))

    def conv_layer(input_, out_channels, ks=3, s=1):
        return lrelu(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))

    def last_layer(input_, out_channels, ks=4, s=1):
        return layers.Dense(units=out_channels)(conv2d(input_, out_channels, ks=ks, s=s))

    inputs = tf.keras.Input(shape=image_shape)
    l1 = first_layer(inputs, options.df_dim, ks=4, s=2)
    l2 = conv_layer(l1, options.df_dim * 2, ks=4, s=2)
    l3 = conv_layer(l2, options.df_dim * 4, ks=4, s=2)
    l4 = conv_layer(l3, options.df_dim * 8, ks=4, s=1)
    l5 = last_layer(l4, options.img_channel, ks=4, s=1)

    model = tf.keras.Model(inputs=inputs, outputs=l5, name=name)
    return model


def generator(image_shape, options, name="generator"):
    def conv_layer(input_, out_channels, ks=3, s=1):
        return layers.ReLU()(batchnorm(conv2d(input_, out_channels, ks=ks, s=s)))

    def gen_module(input_, out_channels, ks=3, s=1):
        ml1 = conv_layer(input_, out_channels, ks, s)
        ml2 = conv_layer(ml1, out_channels, ks, s)
        ml3 = conv_layer(ml2, out_channels, ks, s)
        concat_l = input_ + ml3
        m_out = layers.ReLU()(concat_l)

        return m_out

    inputs = tf.keras.Input(shape=image_shape)
    l1 = conv_layer(inputs, options.gf_dim)
    module1 = gen_module(l1, options.gf_dim)
    module2 = gen_module(module1, options.gf_dim)
    module3 = gen_module(module2, options.gf_dim)
    module4 = gen_module(module3, options.gf_dim)
    module5 = gen_module(module4, options.gf_dim)
    module6 = gen_module(module5, options.gf_dim)
    concate_layer = layers.concatenate([l1, module1,
                                        module2, module3, module4, module5, module6], axis=3)
    concat_conv_l1 = conv_layer(concate_layer, options.gf_dim, ks=3, s=1)
    last_conv_layer = conv_layer(concat_conv_l1, options.glf_dim, ks=3, s=1)
    output = layers.Add(name='output')([conv2d(last_conv_layer, options.img_channel, ks=3, s=1), inputs])

    model = tf.keras.Model(inputs=inputs, outputs=output, name=name)
    return model


# network components
def lrelu(x, leak=0.2):
    return layers.LeakyReLU(alpha=leak)(x)


def batchnorm(input_, name="batch_norm"):
    return layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1,
                                     gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(input_, training=True)


def conv2d(batch_input, out_channels, ks=4, s=2):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return layers.Conv2D(out_channels, kernel_size=ks, strides=s, padding="valid",
                         kernel_initializer=tf.random_normal_initializer(0, 0.02))(padded_input)


#### loss
def least_square(A, B):
    return tf.math.reduce_mean((A - B) ** 2)


def cycle_loss(A, F_GA, B, G_FB, lambda_):
    return lambda_ * (tf.math.reduce_mean(tf.math.abs(A - F_GA)) + tf.math.reduce_mean(tf.math.abs(B - G_FB)))


def identity_loss(A, G_B, B, F_A, gamma):
    return gamma * (tf.math.reduce_mean(tf.math.abs(G_B - B)) + tf.math.reduce_mean(tf.math.abs(F_A - A)))
