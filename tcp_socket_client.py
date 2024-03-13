# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:12:13 2024

@author: Roma
"""
import json
import socket

from scapy.all import *
from scapy.all import sniff, wrpcap

host = 'localhost'  # Server's hostname or IP address
port = 4840         # Server's port 

# Define the packet capture filter
# For example, capturing IP traffic to and from a specific IP and port
filter = "ip host 127.0.0.1 and tcp port 4840"

# Capture packets
packets = sniff(filter=filter, count=1000)  # Adjust count as needed


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


    
with open('bytes_list.bin', 'rb') as file:
    data_list = [line.rstrip(b'\n') for line in file]
send_list_elements(host, port, data_list)

# Save the captured packets to a .pcap file
file_name = "captured_traffic.pcap"
wrpcap(file_name, packets)


