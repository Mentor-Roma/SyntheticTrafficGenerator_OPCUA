# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:15:53 2024

@author: Roma
"""

import socket

host = 'localhost'
port = 4840

def start_server(host, port):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the port
    server_address = (host, port)
    print(f"Starting up on {server_address}")
    sock.bind(server_address)
    
    # Listen for incoming connections
    sock.listen(1)
    
    while True:
        print("Waiting for a connection")
        connection, client_address = sock.accept()
        
        try:
            print(f"Connection from {client_address}")
            
            # Receive the data in small chunks and retransmit it
            while True:
                data = connection.recv(2048)
                if data:
                    print("I have received some data")
                    response = "ACK".encode()
                    connection.sendall(response)
                else:
                    break
            
        finally:
            # Clean up the connection
            connection.close()


    
start_server(host, port)
