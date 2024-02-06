#!/bin/bash
source ~/.bashrc

gnome-terminal -t "launcher" -x bash -c "source ~/.bashrc;source /opt/ros/noetic/setup.bash;cd ~/桌面/follow/arx5_follow2;source ~/桌面/follow/arx5_follow2/devel/setup.bash;roslaunch arm_control arx5v.launch;exec bash;"





