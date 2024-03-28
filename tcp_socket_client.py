# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:12:13 2024

@author: Roma
"""

import socket
import os

from scapy.all import *



host = 'localhost'  # Server's hostname or IP address
port = 4840         # Server's port 

# Define the packet capture filter
# For example, capturing IP traffic to and from a specific IP and port
filter = "ip host 127.0.0.1 and tcp port 4840"

# Capture packets
#packets = sniff(filter=filter, count=1000)  # Adjust count as needed


def send_list_elements(host, port, messages):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect the socket to the server
        server_address = (host, port)
        print(f"Connecting to {server_address}")
        sock.connect(server_address)


        # Send each message in the list
        for message in messages:
            print(f"Sending message: {message}")
            sock.sendall(message)
            # Optionally wait for the server's response
            response = sock.recv(2048)
            print(f"Received: {response.decode()}")
    
    finally:
        print("Closing socket")
        sock.close()


  
with open(r'bytes_list_512.bin', 'rb') as file:
    data_list_512 = [line.rstrip(b'\n') for line in file]
with open(r'bytes_list_656.bin', 'rb') as file:
    data_list_656 = [line.rstrip(b'\n') for line in file]
with open(r'bytes_list_792.bin', 'rb') as file:
    data_list_792 = [line.rstrip(b'\n') for line in file]
with open(r'bytes_list_824.bin', 'rb') as file:
    data_list_824 = [line.rstrip(b'\n') for line in file]



def combine_lists(L0, L1, L2, L3, L4):
    L = []  # Initialize the new list L.
    L1_index, L2_index, L3_index, L4_index = 0, 0, 0, 0  # Initialize indices for each list.

    for value in L0:
        if value == 0:
            # If the next element of L0 is 0, append an element from L1 and then from L2.
            if L1_index < len(L1):  # Check if there are enough elements in L1.
                L.append(L1[L1_index])
                L1_index += 1  # Move to the next index in L1.
            if L2_index < len(L2):  # Check if there are enough elements in L2.
                L.append(L2[L2_index])
                L2_index += 1  # Move to the next index in L2.
        elif value == 1:
            # If the next element of L0 is 1, append an element from L3 and then from L4.
            if L3_index < len(L3):  # Check if there are enough elements in L3.
                L.append(L3[L3_index])
                L3_index += 1  # Move to the next index in L3.
            if L4_index < len(L4):  # Check if there are enough elements in L4.
                L.append(L4[L4_index])
                L4_index += 1  # Move to the next index in L4.

    return L

def combine_lists_v2(L0, L1, L2):
    L = []  # Initialize the new list L.
    L1_index, L2_index = 400, 400  # Initialize indices for L1 and L2.

    for value in L0:
        if value == 0:
            # If the next element of L0 is 0, append an element from L1.
            if L1_index < len(L1):  # Check if there are enough elements in L1.
                L.append(L1[L1_index])
                L1_index += 1  # Move to the next index in L1.
        elif value == 1:
            # If the next element of L0 is 1, append an element from L2.
            if L2_index < len(L2):  # Check if there are enough elements in L2.
                L.append(L2[L2_index])
                L2_index += 1  # Move to the next index in L2.

    return L



def combine_and_slice_lists(L1, L2, L3, index0, index1, index2):
    """
    Form a new list by:
    1. Taking elements of L1 up to index1.
    2. Appending all elements of L2.
    3. Appending elements of L1 from index1 up to index2.
    4. Appending all elements of L3.
    """
    # Ensure index2 is not less than index1
    if index2 < index1:
        print("Error: index2 must be greater than or equal to index1")
        return []

    # Step 1: Elements of L1 up to index1
    new_list = L1[index0:index1]
    
    # Step 2: Append all elements of L2
    new_list += L2
    
    new_index = index1 + 200
    # Step 3: Append elements of L1 from new_index to index2
    new_list += L1[new_index:index2]
    
    # Step 4: Append all elements of L3
    new_list += L3
    
    #Step5: Append remaining elements of L1
    
    new_list += L1[index2:index2+50]
    
    return new_list




# Define the input texts
text_part1 = "covertchannelisa"
text_part2 = "stealthyattack"

# Function to encode a string into binary, concatenate the bits, split into a list of bits, and convert each bit to an integer
def text_to_binary_list_int(text):
    # Encode the text to ASCII
    ascii_encoded_text = text.encode('ascii')
    # Convert each byte to its binary representation, filling to make each byte representation 8 bits long
    binary_representation = [f"{byte:08b}" for byte in ascii_encoded_text]
    # Concatenate the binary values into a single string
    combined_bits = ''.join(binary_representation)
    # Split the combined string into a list of individual bits and convert each to an integer
    list_of_bits = [int(bit) for bit in combined_bits]
    return list_of_bits

# Convert texts to binary and get lists of bits
list_of_bits_part1 = text_to_binary_list_int(text_part1)
list_of_bits_part2 = text_to_binary_list_int(text_part2)


attack_list = combine_lists(list_of_bits_part1, data_list_824, data_list_512, data_list_792, data_list_656)
attack_list2 = combine_lists_v2(list_of_bits_part2, data_list_792, data_list_656)


data_list = combine_and_slice_lists(data_list_792, attack_list, attack_list2, 100, 200, 500)



send_list_elements(host, port, data_list)




