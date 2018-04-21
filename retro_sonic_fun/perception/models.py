import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

from retro_sonic_fun.common.utils import NN


class VAE(metaclass=NN):
    def __init__(self, **params):
        self._name = 'VAE'
        self.input_dim = params['input_dim']
        self.conv_filters = params['conv_filters']
        self.conv_kernel_sizes = params['conv_kernel_sizes']
        self.conv_strides = params['conv_strides']
        self.conv_activations = params['conv_activations']
        self.dense_size = params['dense_size']
        self.conv_t_kernel_sizes = params['conv_t_kernel_sizes']
        self.conv_t_filters = params['conv_t_filters']
        self.conv_t_strides = params['conv_t_strides']
        self.conv_t_activations = params['conv_t_activations']
        self.z_dim = params['z_dim']
        self.vae, self.encoder, self.decoder = self._build()

    def _build(self):

        ###########################################################
        #TF nodes
        ###########################################################

        vae_x = Input(shape=self.input_dim)
        vae_c = [vae_x]
        for _ in range(len(self.conv_filters)):
            vae_c.append(
                Conv2D(
                    filters=self.conv_filters[_],
                    kernel_size=self.conv_kernel_sizes[_],
                    strides=self.conv_strides[_],
                    activation=self.conv_activations[_])(vae_c[_]))

        vae_z_in = Flatten()(vae_c[-1])
        vae_z_mean = Dense(self.z_dim)(vae_z_in)
        vae_z_log_var = Dense(self.z_dim)(vae_z_in)
        vae_z = Lambda(self.sampling)([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(self.z_dim, ))

        vae_dense = Dense(self.dense_size)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1, 1, self.dense_size))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_c_t = [vae_z_out_model]
        vae_c_t_f = []
        for _ in range(len(self.conv_t_filters)):
            vae_c_t_f.append(
                Conv2DTranspose(
                    filters=self.conv_t_filters[_],
                    kernel_size=self.conv_t_kernel_sizes[_],
                    strides=self.conv_t_strides[_],
                    activation=self.conv_t_activations[_]))

            vae_c_t.append(vae_c_t_f[_](vae_c_t[_]))

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_c_t_decoder = [vae_z_out_decoder]
        for _ in range(len(self.conv_t_filters)):
            vae_c_t_decoder.append(vae_c_t_f[_](vae_c_t_decoder[_]))

        ###########################################################
        #Losses and Models
        ###########################################################
        def vae_r_loss(y_true, y_pred):

            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        def vae_kl_loss(y_true, y_pred):
            return -0.5 * K.mean(
                1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var),
                axis=-1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

        vae = Model(vae_x, vae_c_t[-1])
        vae.compile(
            optimizer='rmsprop',
            loss=vae_loss,
            metrics=[vae_r_loss, vae_kl_loss])
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_c_t_decoder[-1])

        return vae, vae_encoder, vae_decoder

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], self.z_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    @property
    def name(self):
        return self._name


if __name__ == '__main__':

    from retro_sonic_fun.common.utils import create_json_params,\
     load_json_params

    # params = {
    #     'input_dim': (64, 64, 3),
    #     'conv_filters': [32, 64, 64, 128],
    #     'conv_kernel_sizes': [4, 4, 4, 4],
    #     'conv_strides': [2, 2, 2, 2],
    #     'conv_activations': ['relu', 'relu', 'relu', 'relu'],
    #     'dense_size': 1024,
    #     'conv_t_kernel_sizes': [64, 64, 32, 3],
    #     'conv_t_filters': [5, 5, 6, 6],
    #     'conv_t_strides': [2, 2, 2, 2],
    #     'conv_t_activations': ['relu', 'relu', 'relu', 'sigmoid']
    # }

    # print(vars(g))
    # create_json_params(params, 'VAE')
    parameters = load_json_params('VAE.json')
    vae = VAE(**parameters)
    print(vars(vae))
