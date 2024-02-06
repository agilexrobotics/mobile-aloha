#!/bin/bash
source ~/.bashrc
workspace=${HOME}/code/aloha/follow

gnome-terminal -t "launcher" -- bash -c "source ${HOME}/.aloha_camera_config.bash;"
sleep 1
gnome-terminal -t "launcher" -- bash -c "source ~/.bashrc;source /opt/ros/${ROS_DISTRO}/setup.bash;cd ${workspace}/realsense_ws;source ${workspace}/realsense_ws/devel/setup.bash;roslaunch realsense2_camera aloha.launch; exec bash;"
