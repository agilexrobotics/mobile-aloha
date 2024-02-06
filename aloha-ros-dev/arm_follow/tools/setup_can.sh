#!/bin/bash

# Set up can0
while [ 1 ]
do
sudo ip link set up can0 type can bitrate 1000000
sudo ip link set up can1 type can bitrate 1000000
sudo ip link set up can2 type can bitrate 1000000
sudo ip link set up can3 type can bitrate 1000000
sleep 1
done
