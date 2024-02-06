#!/bin/bash
source ~/.bashrc
# workspace=${HOME}/follow
# workspace=${HOME}/Documents/share/follow
workspace=${HOME}/aloha-ros-dev/arm_follow
password=1

echo workspace:${workspace}  password: ${password}
echo ${ROS_DISTRO}

# 1 启动roscore
gnome-terminal -t "roscore" -- bash -c "source ~/.bashrc;source /opt/ros/${ROS_DISTRO}/setup.bash;roscore;exec bash;"
sleep 1

# 2 ip配置 can口
gnome-terminal -t "can" -- bash -c "source ~/.bashrc; source /opt/ros/${ROS_DISTRO}/setup.bash; echo ${password} | sudo -S ${workspace}/tools/setup_can.sh; candump can0; exec bash;"

# 3 启动臂
gnome-terminal -t "launcher" -- bash -c "source ~/.bashrc;source /opt/ros/${ROS_DISTRO}/setup.bash;cd ${workspace}/arx5_follow;source ${workspace}/arx5_follow/devel/setup.bash;roslaunch arm_control arx5v.launch; exec bash;"
gnome-terminal -t "launcher" -- bash -c "source ~/.bashrc;source /opt/ros/${ROS_DISTRO}/setup.bash;cd ${workspace}/arx5_follow2;source ${workspace}/arx5_follow2/devel/setup.bash;roslaunch arm_control arx5v.launch; exec bash;"
gnome-terminal -t "launcher" -- bash -c "source ~/.bashrc;source /opt/ros/${ROS_DISTRO}/setup.bash;cd ${workspace}/arx5_follow3;source ${workspace}/arx5_follow3/devel/setup.bash;roslaunch arm_control arx5v.launch; exec bash;"
gnome-terminal -t "launcher" -- bash -c "source ~/.bashrc;source /opt/ros/${ROS_DISTRO}/setup.bash;cd ${workspace}/arx5_follow4;source ${workspace}/arx5_follow4/devel/setup.bash;roslaunch arm_control arx5v.launch; exec bash;"

# gnome-terminal -t "launcher" -- bash -c "roslaunch realsense2_camera rs_multiple_devices.launch;exec bash;"



