#!/bin/bash
source ~/.bashrc
gnome-terminal -t "roscore" -x bash -c "source ~/.bashrc;source /opt/ros/noetic/setup.bash;roscore;exec bash;"
sleep 1
gnome-terminal -t "can" -x bash -c "source ~/.bashrc;source /opt/ros/noetic/setup.bash;echo 123 | sudo -S ./setup_can.sh;candump can0;exec bash;"
gnome-terminal -t "launcher" -x bash -c "source ~/.bashrc;source /opt/ros/noetic/setup.bash;cd ~/桌面/follow/arx5_follow;source ~/桌面/follow/arx5_follow/devel/setup.bash;roslaunch arm_control arx5v.launch;exec bash;"





