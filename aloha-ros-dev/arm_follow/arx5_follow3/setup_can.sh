#!/bin/bash

# Set up can2
while [ 1 ]
do
sudo ip link set up can2 type can bitrate 1000000
sudo ip link set up can1 type can bitrate 1000000
sudo ip link set up can2 type can bitrate 1000000
sudo ip link set up can2 type can bitrate 1000000
sleep 1
done
