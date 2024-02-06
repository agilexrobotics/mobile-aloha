
# Mobile ALOHA

A Low-cost Open-source Hardware System for Bimanual Teleoperation

# 1 ENV

1. 默认ubuntu20.04，ros1-noetic版本, python3.8, torch-1.11.0+cu113
2. [ACT环境配置](doc/env.md)
3. [机械臂环境配置](https://github.com/agilexrobotics/mobile-aloha/blob/devel/aloha-ros-dev/readme.md)

# 2 生成自定义数据集

~~~python
# 1 启动roscore
roscore

# 2 采集数据
## 2.1 进入aloha-ros-dev目录
cd aloha-ros-dev
## 2.2 采集数据
python scripts/record_data.py --task_name aloha_mobile_dummy --max_timesteps 500 --dataset_dir ./data --episode_idx 0

# 3 可视化数据
python scripts/visualize_episodes.py --episode_idx 0
~~~

# 3 训练
~~~python
python act/train.py --task_name aloha_mobile_dummy --ckpt_dir ckpt --chunk_size 30
~~~

# 4 推理
+ 只启动从臂
~~~python
python act/inference.py --ckpt_dir ckpt --task_name aloha_mobile_dummy
~~~


# 5 报错汇总

~~~python
# 1 ModuleNotFoundError: No module named 'aloha_scripts'
export PYTHONPATH=$PYTHONPATH:"./"
# 或者 python源码中添加
import sys  # 添加python环境
sys.path.append("./")


# 2 如果报util错误重新安装detr


# 3 Unable to register with master node [http://localhost:11311]: master may not be running yet. Will keep trying.
启动roscore即可

# 4 训练报错 数据集路径不对
# FileNotFoundError: [Errno 2] Unable to open file (unable to open file: name = '/home/lin/mobile-aloha/data/aloha_mobile_dummy/episode_0.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
修改aloha_scripts/constants.py文件中TASK_CONFIGS参数即可
~~~

# 6 数据集格式  

+ TimeStep、action

~~~python
image_dict = dict()
for cam_name in camera_names: 
    image_dict[cam_name] = image

# 观察状态
import dm_env, collections
obs = collections.OrderedDict()  # 有序的字典

# qpos qvel effort 14维度 base_vel 2维度
obs['qpos'] = np.random.rand(14)
obs['qvel'] = np.random.rand(14)
obs['effort'] = np.random.rand(14)
obs['images'] = image_dict
obs['base_vel'] = np.array([0.1, 0.3])

# 初始状态的first
ts = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation=obs)
~~~