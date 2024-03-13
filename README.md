# SyntheticTrafficGenerator_OPCUA
This repository hosts an ML-based synthetic OPC UA traffic generator.
# Synthetic OPC UA Traffic Generator

This repository hosts an ML-based synthetic OPC UA traffic generator.

## Installation

1. Clone the repository: `git clone https://github.com/Mentor-Roma/SyntheticTrafficGenerator_OPCUA.git`
2. Navigate into the cloned repository: `cd SyntheticTrafficGenerator_OPCUA`
3. Create a virtual environment `python3 -m venv venv`
4. Activate the virtual environment `source venv/bin/activate`  # On Windows use `venv\Scripts\activate`
5. Install all necessary requirements `pip install -r requirements.txt`

## Build, Train and Generate

Run the `build_train_generate.py` script when you want to build, compile and train the model by yourself, on same or different OPC UA traffic. Note that this training process will take some time depending on the processing capabilities or not (GPU, TPU...)

The script does the following:

1. Read the .pcap files, process it into suitable format for ML model

2. Build and compile the VAE-LSTM hybrid model

3. Train the model

4. Generate a specific number (defined by variable `num_samples`) of packet payload

5. Save these payloads as byte objects in a binary file (called `bytes_list.bin`) that will be used by the socket program

##Generate
Run the `generate.py` script when you do not want to train the model by yourself. 

