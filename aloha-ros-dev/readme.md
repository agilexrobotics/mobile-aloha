

# 1 机械臂启动配置

1. 依赖安装
~~~python
sudo apt install libkdl-parser-dev
~~~


2. 先断电再断can，再接can最后上电
+ 13主臂， i进入键盘模式， 24从臂

+ `/home/lin/follow/arx5_follow/src/arm_control/src/App/arm_control.cpp`文件中
`modifyLinkMass(model_path, model_path, 0.640)` 质量

3. 编译工程
~~~python
./tools/build.sh
~~~

4. 运行

~~~python
# 1 修改配置follow.sh文件
# 1.1 工程目录
workspace=￥{HOME}/test/follow
# 1.2 密码
password=1

# 2 运行
./tools/start.sh
~~~


5. 删除编译文件
~~~python
./tools/remove_make_file.sh
~~~


# 2 相机配置

## 2.1 realsense相机配置

### 2.1.1 librealsense驱动安装
~~~python
# 1 相机依赖
sudo apt-get install libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libuvc-dev libgoogle-glog-dev

# 2. 添加规则库
cd realsense2_camera_ws/thirdparty
tar -xzvf librealsense-2.50.0.tar.gz
cd librealsense-2.50.0
./scripts/setup_udev_rules.sh

## 2.1 该步骤可以跳过不执行。 构建和应用补丁内核模块，下面的脚本将下载、修补和构建受 realsense 影响的内核模块（驱动程序）。然后它将尝试插入修补模块而不是活动模块。如果失败，原始 uvc 模块将被恢复。
./scripts/patch-realsense-ubuntu-lts.sh
 
# 3. 编译安装
mkdir build && cd build && cmake ..
sudo make install
~~~

### 2.1.2 realsense-ros功能包 

1. 生成相机的唯一序列号
+ 生成相机序列号配置文件, 执行一次即可
+ 如果相机更换了， 需要重新操作该步骤，更换对应的相机序列号

~~~python
# 1 依赖安装
sudo apt-get install ros-$ROS_DISTRO-ddynamic-reconfigure ros-$ROS_DISTRO-rgbd-launch

# 2 编译源码
cd realsense2_camera_ws
catkin_make

# 3 生成序列号
# 3.1 生成序列号
./devel/lib/realsense2_camera/list_devices_device
# 将生成的Serial number序列号填入realsense2_camera/launch/目录下rs_multiple_devices.launch文件serial_no_camera*的默认值即可, 这里对rs_multiple_devices.launch复制后重命名为aloha.launch进行操作

# 3.2 自动配置序列号文件 （可以跳过，直接看启动相机）
rosrun realsense2_camera list_devices_device && chmod +x ~/.aloha_camera_config.bash
# 配置文件打开终端自动生效
echo "source ~/.aloha_camera_config.bash" >> ~/.bashrc
# 生效相机序列号 
source ~/.aloha_camera_config.bash
~~~

2. 启动相机
~~~python
# 生效工作空间
source devel/setup.bash

# 2 启动相机
roslaunch realsense2_camera aloha.launch
~~~

## 2.2 ros_astra_camera

~~~python
# 1 下载ros_astra_camera
git clone https://github.com/orbbec/ros_astra_camera.git
# 加代理下载
git clone https://mirror.ghproxy.com/https://github.com/orbbec/ros_astra_camera.git
# 本工程自带相机ros驱动源码, 直接进入工作空间, 进行下面操作即可
cd astra_camera_ws

# 2 依赖安装
sudo apt install libgflags-dev  ros-$ROS_DISTRO-image-geometry ros-$ROS_DISTRO-camera-info-manager ros-$ROS_DISTRO-image-transport ros-$ROS_DISTRO-image-publisher libgoogle-glog-dev libusb-1.0-0-dev libuvc-dev libeigen3-dev

# 3 编译相机包
catkin_make

# 4 设置规则
source devel/setup.bash && rospack list
roscd astra_camera
./scripts/create_udev_rules
sudo udevadm control --reload && sudo  udevadm trigger

# 5 接上相机usb数据线,运行ls /dev/video*
ls /dev/video*
# 终端出现：/dev/video0  /dev/video1  /dev/video2  /dev/video3  /dev/video4  /dev/video5
# 一个相机会显示2个/dev/video*, 这里显示的6个，即3个相机

# 6 生成相机的唯一序列号
./devel/lib/astra_camera/list_devices_node
# 3个相机会输出3个不同的Serial number,将3个Serial number填入camera_ws/src/ros_astra_camera/launch/multi_dabai.launch中的camera*_serila_number参数即可，这里将multi_dabai.launch复制了一份重命名为aloha.launch

# 6 启动相机程序
source devel/setup.bash
roslaunch astra_camera aloha.launch

# 7 查看相机图像
rqt_image_view
# 如果rqt启动报错, 请关闭conda虚拟环境后，重新执行rqt_image_view， 如下
conda deactivate && rqt_image_view
~~~


# 3 采集训练数据

1. 依赖安装
~~~python
pip install h5py dm_env  argparse numpy==1.23.4 tqdm==4.66.1
~~~


2. 录制bag包
~~~python
# 1 图像压缩格式
rosbag record -o test.bag /camera1/color/image_raw/compressed /camera2/color/image_raw/compressed /camera3/color/image_raw/compressed /master/joint_left /master/joint_right /puppet/joint_left /puppet/joint_right

# 2 图像非压缩格式
rosbag record -o test.bag /camera1/color/image_raw /camera2/color/image_raw /camera3/color/image_raw /master/joint_left /master/joint_right /puppet/joint_left /puppet/joint_right
~~~


3. 录制训练数据

+ ros消息多机通讯是通过无线传输可以，图像可以采用CompressedImage压缩格式
+ 如果是有线连接通讯建议图像直接用Image格式, 数据转换中耗时更少
+ 结束当前回合按`ctrl +c` 即可

~~~python
# 1 启动roscore
roscore

# 2 采集数据
## 2.1 进入aloha-ros-dev目录
cd aloha-ros-dev
## 2.2 采集数据
python scripts/record_data.py --task_name aloha_mobile_dummy --max_timesteps 500 --dataset_dir ./data --episode_idx 0
~~~

4. 可视化数据
~~~python
python scripts/visualize_episodes.py --episode_idx 0
~~~
