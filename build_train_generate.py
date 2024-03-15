# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:13:26 2024
@author: Roma


1. Read the .pcap files, process it into suitable format for ML model

2. Build and compile the VAE-LSTM hybrid model

3. Train the model

4. Generate a specific number (defined by variable `num_samples`) of packet payload

5. Save these payloads as byte objects in a binary file (called `bytes_list.bin`) that will be used by the socket program 


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



# Convert packet payload to a bit sequence (string of 0s and 1s)
def packet_to_bits(packet):
    return ''.join(format(byte, '08b') for byte in bytes(packet))

# Load the pcap file
packets = rdpcap("real_opcua_packets.pcapng")

# List to hold bit sequences of OPC UA payloads
opcua_bit_sequences = []

for packet in packets:
    if TCP in packet and packet[TCP].dport == 4840 or packet[TCP].sport == 4840:  # Default OPC UA port
        payload = packet[TCP].payload
        if payload:
            bit_sequence = packet_to_bits(payload)
            opcua_bit_sequences.append(bit_sequence)

# opcua_bit_sequences now contains the bit sequences of OPC UA payloads
# You can print them or perform further analysis
#print(opcua_bit_sequences[0])

bits_message = opcua_bit_sequences
#Get the lengths of each payload message
message_lengths = [len(message) for message in bits_message]
# Use Counter to count the frequency of each length
frequency = Counter(message_lengths)
truncated_bits_message = [message for message in bits_message if len(message) == 656]
#Get the lengths of each payload message
message_lengths = [len(message) for message in truncated_bits_message]
# Use Counter to count the frequency of each length
frequency = Counter(message_lengths)
# Print the frequency of each number
for number, count in frequency.items():
    print(f'Number {number} appears {count} times.')

# Convert each bit string to a list of integers
bit_lists = [[int(bit) for bit in string] for string in truncated_bits_message]

# Create a 2D NumPy array from the list of lists of integers
truncated_bits_message = np.array(bit_lists)
epochs = 100
input_dimension = truncated_bits_message.shape[1]
timesteps = 5
batch_size = 64
latent_dim = 16
num_sequences = len(truncated_bits_message)

# Pad the data to ensure the last sequences can continue from the beginning
padded_data = np.concatenate([truncated_bits_message[-timesteps+1:, :], truncated_bits_message, truncated_bits_message[:timesteps-1, :]])
# Initialize X with zeros
X_train = np.zeros((num_sequences, timesteps, truncated_bits_message.shape[1]))
# Populate X with sequences
for i in range(num_sequences):
    X_train[i, :, :] = padded_data[i:i+timesteps, :]


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

from keras.callbacks import EarlyStopping

# Assuming X_train is your training data and already normalized
# X_train shape: (num_samples, timesteps, features)


# Callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')

history = vae.fit(
    X_train,  X_train, # VAEs are trained to reconstruct their input
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,  # Adjust the validation split as needed
    callbacks=[early_stopping]
)


# Extract loss and validation loss from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a range for the number of epochs
x_axis = range(1, len(train_loss) + 1)

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(x_axis, train_loss, 'bo-', label='Training Loss')
plt.plot(x_axis, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("Train_and_val_loss.pdf")


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


plt.figure(figsize=(15,5))
plt.plot(generated_data[0], 'bo', label='Generated data', markersize=2)
plt.plot(generated_data[49], 'ro', label='Generated data', markersize=2)
plt.plot(truncated_bits_message[25], 'yo', label='Real data', markersize=2)
plt.xlabel('Features')
plt.ylabel('values')
plt.title('Generated and realistic data samples')
plt.legend()
plt.savefig("overview_generated_stream.pdf")

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
        


