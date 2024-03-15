# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:13:26 2024
@author: Roma




1. Generate a specific number (defined by variable `num_samples`) of packet payload

2. Save these payloads as byte objects in a binary file (called `bytes_list.bin`) that will be used by the socket program 


"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

from tensorflow.keras import layers
from keras.layers import Input, LSTM, Dense, Lambda, RepeatVector, TimeDistributed
from keras.models import Model
from sklearn.metrics import mean_squared_error

from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from scapy.all import rdpcap, TCP, IP, Ether, wrpcap



num_samples = 1000  # Number of time series sequences to generate




epochs = 100
input_dimension = 656
timesteps = 5
batch_size = 64
latent_dim = 16
num_sequences = 2550



# Encoder
inputs = Input(shape=(timesteps, input_dimension))
x = LSTM(256, return_sequences=True)(inputs)  # Example LSTM layer, adjust the number of units as needed
x = LSTM(128, return_sequences=True)(x)
x = LSTM(64, return_sequences=True)(x)
x = LSTM(32)(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#encoder.summary()


# Decoder
latent_inputs = Input(shape=(latent_dim,))
x = RepeatVector(timesteps)(latent_inputs)
x = LSTM(32, return_sequences=True)(x)  # Match the LSTM units with the encoder
x = LSTM(64, return_sequences=True)(x)
x = LSTM(128, return_sequences=True)(x)
x = LSTM(256, return_sequences=True)(x)
outputs = TimeDistributed(Dense(input_dimension, activation='sigmoid'))(x)

# Decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
#decoder.summary()

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_lstm')

# Add VAE loss
reconstruction_loss = tf.keras.losses.mean_squared_error(K.flatten(inputs), K.flatten(outputs))
#reconstruction_loss *= timesteps * input_dimension
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.load_weights("model_weights.h5")


#Generating new paylaod to be used in packets



# Sample random points in the latent space
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
generated_data = decoder.predict(random_latent_vectors)
def post_process(message):
    message = np.array(message)
    #Applying threshold to binarize the images
    message[message > 0.4] = 1
    message[message <= 0.4] = 0
    
    return message

generated_data = generated_data.reshape(-1, generated_data.shape[2])
generated_data = post_process(generated_data)#Binarize the generated data with the post_process function


def bits_to_messages(bit_sequences):
    messages = []
    for bit_sequence in bit_sequences:
        # Convert the list of bits back into a bytes object
        # First, convert the list of integers (bits) back into a string of bits
        bits_str = ''.join(str(int(bit)) for bit in bit_sequence)
        # Then, split the string into chunks of 8 bits
        byte_strings = [bits_str[i:i+8] for i in range(0, len(bits_str), 8)]
        # Convert each chunk of 8 bits into a byte
        message = bytes([int(byte_str, 2) for byte_str in byte_strings])
        messages.append(message)
    return messages

messages = bits_to_messages(generated_data)

with open(r'bytes_list.bin', 'wb') as file:
    for byte_obj in messages:
        file.write(byte_obj + b'\n')  # Adding a newline as a separator
        


