#!/bin/bash
source ~/.bashrc
workspace=${HOME}/code/aloha/follow

gnome-terminal -t "launcher" -- bash -c "source ~/.bashrc;source /opt/ros/${ROS_DISTRO}/setup.bash;cd ${workspace}/realsense_ws;source ${workspace}/realsense_ws/devel/setup.bash;roslaunch realsense2_camera list_camera_device.launch; exec bash;"
sleep 1
gnome-terminal -t "launcher" -- bash -c "chmod +x ${HOME}/.aloha_camera_config.bash;"