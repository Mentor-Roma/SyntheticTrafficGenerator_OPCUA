# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:15:53 2024

@author: Roma
"""

import sys
import socket
import signal
import subprocess

host = 'localhost'
port = 4840

def start_tcpdump_capture(interface, filename):
    # Build the tcpdump command
    cmd = ['tcpdump', '-i', interface, '-w', filename]
    
    # Start tcpdump as a subprocess
    tcpdump_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tcpdump_proc

def stop_tcpdump_capture(tcpdump_proc):
    # Send SIGINT to the tcpdump process to stop it gracefully
    tcpdump_proc.send_signal(signal.SIGINT)
    # Wait for the process to terminate
    tcpdump_proc.wait()
    print("tcpdump capture stopped and saved to file.")


def start_server(host, port):
    
    
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Ensure the socket is closed on exit
    with sock:
        # Bind the socket to the port
        server_address = (host, port)
        print(f"Starting up on {server_address}")
        sock.bind(server_address)
        
        # Listen for incoming connections
        sock.listen(1)
        
        try:
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
        
        except KeyboardInterrupt:
            print("Server is shutting down.")
            sock.close()
            sys.exit(0)  # Optionally exit program
        

start_server(host, port)