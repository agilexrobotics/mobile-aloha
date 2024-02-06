import os
import time
import h5py
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm
import cv2

import dm_env

# 机器人的参数
from constants import DT, START_ARM_POSE, TASK_CONFIGS, FPS

from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
# from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
# from robot_utils import move_arms, torque_on, torque_off, move_grippers
# from real_env import make_real_env, get_action

# from real_env import RealEnv

# from interbotix_xs_modules.arm import InterbotixManipulatorXS

import IPython
e = IPython.embed


from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageRecorder:
    def __init__(self) -> None:
        pass


import rospy

rospy.init_node('record_episodes', anonymous=True)




bridge = CvBridge()

class Img:
    def __init__(self) -> None:
        self.imgl = None
        self.imgr = None
        self.img_top = None

class Arm:
    def __init__(self) -> None:
        self.action = None
        self.pose = None
        self.vel = None
        self.effort = None

class BaseRobot:
    def __init__(self) -> None:
        self.linear_vel = None
        self.angular_vel = None

img = Img()
class Arm:
    pass
master_armL, master_armR, puppet_armL, puppet_armR = [Arm() for _ in range(4)]

base_robot = BaseRobot()

def imageL_callback(msg):
    img.imgl = bridge.imgmsg_to_cv2(msg, 'passthrough')

def imageR_callback(msg):
    img.imgr = bridge.imgmsg_to_cv2(msg, 'passthrough')

def imageTop_callback(msg):
    img.img_top = bridge.imgmsg_to_cv2(msg, 'passthrough')

def masterL_callback(msg):
    master_armL.vel = msg.velocity
    master_armL.pose = msg.position
    master_armL.effort = msg.effort
    master_armL.action = msg.position

def masterR_callback(msg):
    master_armR.vel = msg.velocity
    master_armR.pose = msg.position
    master_armR.effort = msg.effort
    master_armR.action = msg.position

def puppetL_callback(msg):
    puppet_armL.vel = msg.velocity
    puppet_armL.pose = msg.position
    puppet_armL.effort = msg.effort
    puppet_armL.action = msg.position

def puppetR_callback(msg):
    puppet_armR.vel = msg.velocity
    puppet_armR.pose = msg.position
    puppet_armR.effort = msg.effort
    puppet_armR.action = msg.position

rospy.Subscriber('/image_left', Image, imageL_callback, queue_size=1)
rospy.Subscriber('/image_right', Image, imageR_callback, queue_size=1)
rospy.Subscriber('/image_top', Image, imageTop_callback, queue_size=1)

rospy.Subscriber('/master_left', JointState, masterL_callback, queue_size=1)
rospy.Subscriber('/master_right', JointState, masterR_callback, queue_size=1)
rospy.Subscriber('/puppet_right', JointState, puppetL_callback, queue_size=1)
rospy.Subscriber('/puppet_left', JointState, puppetR_callback, queue_size=1)

rospy.spin()



def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors
    
    # interbotix_xs_modules/src/interbotix_xs_modules/arm.py
   
    # 重启从臂
    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    
    # 设置从臂的模式： 组 取名arm position  这里起的是个服务  OperatingModes.srv
    puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    
    # 设置从爪的模式： 单个控制 取名gripper current_based_position
    puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    
    # 主臂同理配置
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
    
    # 右边也配置
    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    # TorqueEnable 扭矩服务  TorqueEnable.srv
    # torque_on执行的下面2个函数
    # bot.dxl.robot_torque_enable("group", "arm", True)
    # bot.dxl.robot_torque_enable("single", "gripper", True)
    
    # 打开扭矩
    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]  
    
    # 移动臂初始设置的末端  bot.arm.set_joint_positions设置50个插值并发布set_joint_positions  时间1.5秒完成
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)
    # move grippers to starting position
    
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)


    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    # 禁用爪的扭矩，以允许用户移动
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    
    print(f'Close the gripper to start')
    close_thresh = -1.4
    pressed = False  # False夹爪不处于被压状态  已经打开

    while not pressed:
        # 获取主夹爪的位置
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        
        # 如果位置满足关的阈值 被压状态 = Ture
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT/10)
    
    # 关闭 pressed = True 主臂和主夹爪关闭扭矩
    # bot.dxl.robot_torque_enable("group", "arm", False)
    # bot.dxl.robot_torque_enable("single", "gripper", False)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    
    print(f'Started!')

# 计算一个回合  
# 时间间隔, 最大steps, 相机名称, 保存路径, 数据名称, 重写
def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    print(f'Dataset name: {dataset_name}')
    
    # dt 时间间隔 0.02
    # source of data  
    
    # 机械臂模型  加载模型 机械臂组名 夹爪名
    '''
        InterbotixManipulatorXS类 初始化
        7个服务: 操作模式 设置扭矩状态 重启 机械臂信息(关节信息) MotorGains, set和get(RegisterValues)
        4个话题: JointGroupCommand JointSingleCommand JointTrajectoryCommand JointState
    
    '''
    
    # master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
    #                                           robot_name=f'master_left', init_node=True)
    
    # master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
    #                                            robot_name=f'master_right', init_node=False)
    
    
    # 定义了ros订阅器  定义从臂和从爪puppet_bot_left，puppet_bot_right 还有图像数据
    # env = make_real_env(init_node=False, setup_robots=False)
    # 这里定义了2个从臂从爪
    # env = RealEnv(init_node=False, setup_robots=False, setup_base=False)
    
    # saving dataset 保存数据路径
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    dataset_path = os.path.join(dataset_dir, dataset_name)

    # 如果文件存在和不需要从重写就退出
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    # 移动4个机器到一个开始的pose，启动远程操作 等2个被动夹爪闭合
    
    # 1 初始场景 初始4个臂和4个爪以自身基坐标系运动到相同的位置点， 归0
    
    # 1. 设置左右主从臂爪的操作模式(臂group，爪single)->都开启扭矩模式自动运动到起始状态
        # ->取消主爪扭矩模式(才能人操作)-> 主爪运动到处理关闭状态->关闭主臂和主爪的扭矩可供人操作
    
    # opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

    # Data collection  初始状态的时间步

    # ts = env.reset(fake=True)  # fake = true 从夹爪不需要重启 ，将从臂，从夹爪处于开始状态位置
    
    
    # 图像数据
    image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    # imgs = ImageRecorder()

    image_dict = dict()
    for cam_name in camera_names: 
        # setattr(imgs, f'{cam_name}_image', image)
        # image_dict[cam_name] = getattr(imgs, f'{cam_name}_image')
        image_dict[cam_name] = image
    # 观察状态
    import dm_env, collections
    obs = collections.OrderedDict()  # 有序的字典
    
    # 获取重臂的初始状态
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

    # env.reset 返回的是一个强化学习环境中时间步的类。通常包含有关特定时间步的环境状态的信息。
    # 时间步 动作 时间
    timesteps = [ts]   
    actions = []
    # 时间 包括开始时间刻度 获得主臂主爪后时间刻度，从臂从爪运动完后返回时间步状态的时间刻度
    actual_dt_history = []  

    time0 = time.time()
    DT = 1 / FPS   # 时间

    # 累积一个回合的动作  因为是ros-sub格式 action会随时间是变的
    # max_timesteps 时间内的累积 根据 max_timesteps=默认1000循环
    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #
        
        # 1 获取master动作  主master(左右2-臂6-夹爪1) 14个维度
        # action = get_action(master_bot_left, master_bot_right)
        action = np.random.rand(14)
        
        t1 = time.time() #
        
        # 2 获取从臂的状态
        # ts = env.step(action)  # 根据动作 返回中间timeStep
        
        # 图像数据  # getattr获取相机的属性
        '''
            attrs = [
                'acceleration',
                'angular_acceleration',
                'angular_velocity',
                'mapper_confidence',
                'rotation',
                'tracker_confidence',
                'translation',
                'velocity',
                ]
        '''

        # 这里还需要添加从臂跟随action的动作

        image_dict = dict()
        for cam_name in camera_names: 
            # setattr(imgs, f'{cam_name}_image', image)
            # image_dict[cam_name] = getattr(imgs, f'{cam_name}_image')
            image_dict[cam_name] = image
        
        # 观察状态
        import dm_env, collections
        obs = collections.OrderedDict()  # 有序的字典
        
        obs['images'] = image_dict
    
        # 需要获取从臂的状态和底盘的数据
        # qpos qvel effort 14维度 base_vel 2维度
        obs['qpos'] = np.random.rand(14)
        obs['qvel'] = np.random.rand(14)
        obs['effort'] = np.random.rand(14)
        obs['base_vel'] = np.array([0.1, 0.3])
        
        # 中间时间步
        ts = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=None,
            discount=None,
            observation=obs)
        
        t2 = time.time() #
        
        timesteps.append(ts)   # 记录的从 
        actions.append(action)  # 保存动作  记录的主
        actual_dt_history.append([t0, t1, t2])  # 1 初始时间 获取动作后时间 根据动作获得奖励后时间
                    
                    # 0.02 -(小于0.02)   0.02 * 1000 = 20 秒
        time.sleep(max(0, DT - (time.time() - t0)))
    
    print(f'Avg fps: {max_timesteps / (time.time() - time0)}')

    # Torque on both master bots 打开主臂扭矩 停止人工操作
    # torque_on(master_bot_left)
    # torque_on(master_bot_right)
    
    # # Open puppet grippers # 设置从爪为的单一操作模式 
    # env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    
    # # 执行 只操作从爪->设置为打开状态  PUPPET_GRIPPER_JOINT_OPEN = 1.14
    # move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    # 计算时间的频率
    freq_mean = print_dt_diagnosis(actual_dt_history)  # 平均时间算的频率
    
    # 时间均值  如果小于30 就跳过  1/30  耗时小于0.03秒的就不保存为数据集
    if freq_mean < 30:
        print(f'\n\nfreq_mean is {freq_mean}, lower than 30, re-collecting... \n\n\n\n')
        return False

    # 以下是一个一般化的 "TimeStep" 的概述：
    # 1 step_type： 表示时间步的类型。它通常有以下几种取值：

    # FIRST：表示一个新的 episode 的开始。
    # MID：表示 episode 中的中间步骤。
    # LAST：表示 episode 中的最后一步。
    # ONLY：当环境只有一个时间步时使用。
    
    # 2 observation： 表示环境在当前时间步的观测或状态。这通常是一个 NumPy 数组、字典或数组列表，包含了智能体在当前时刻感知到的环境信息。
    # 3 reward： 表示在当前时间步智能体获得的奖励。奖励是一个标量值，表示智能体执行某个动作后环境对其的反馈。
    # 4 discount： 表示折扣因子，用于计算未来奖励的折现值。通常是一个介于 0 到 1 之间的值。在某些情况下，可能为 None，表示没有折扣因子的概念。

    """
    For each timestep:
    observations  观测状态 1 图像  腕关节图像 2 pose(2 * [6 + 1]) vel (2 * [6 + 1])
    - images       
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    
    action                  (14,)         'float64' 
    base_action             (2,)          'float64'
    """
    
    # 数据字典
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
        # '/base_action_t265': [],
    }

    # 相机字典  观察的图像
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)   # 动作  当前动作
        ts = timesteps.pop(0)     # 奖励  前一帧
        
        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        
        # 实际发的action
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])
        
        # 相机数据
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
    
    # plot /base_action vs /base_action_t265
    # import matplotlib.pyplot as plt
    # plt.plot(np.array(data_dict['/base_action'])[:, 0], label='base_action_linear')
    # plt.plot(np.array(data_dict['/base_action'])[:, 1], label='base_action_angular')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 0], '--', label='base_action_t265_linear')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 1], '--', label='base_action_t265_angular')
    # plt.legend()
    # plt.savefig('record_episodes_vel_debug.png', dpi=300)

    COMPRESS = False
    # 是否压缩图像
    if COMPRESS: 
        # JPEG compression
        t0 = time.time()
        # 50是压缩质量 0-100
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
        
        compressed_len = []
        
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        print(f'compression: {time.time() - t0:.2f}s')

        # 图像pad相同的大小
        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print(f'padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()
    
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性： 
        # 1 是否仿真
        # 2 图像是否压缩
        # 
        root.attrs['sim'] = False                   
        root.attrs['compress'] = COMPRESS   
        
        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )            
            
            else:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        

        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        _ = obs.create_dataset('qvel', (max_timesteps, 14))
        _ = obs.create_dataset('effort', (max_timesteps, 14))
        _ = root.create_dataset('action', (max_timesteps, 14))
        _ = root.create_dataset('base_action', (max_timesteps, 2))
        
        # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

        # data_dict写入h5py.File
        for name, array in data_dict.items():   # 名字+值
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            root['/compress_len'][...] = compressed_len

    print(f'Saving: {time.time() - t0:.1f} secs', dataset_path)

    return True


def main(args):
    
    # 根据任务的名字获得参数配置：数据保存路径 场景长度 相机名字
    # 任务名字
    task_config = TASK_CONFIGS[args['task_name']]
    
    # 数据路径
    dataset_dir = task_config['dataset_dir']
    
    # 场景长度 时间累积
    max_timesteps = task_config['episode_len']
    
    # 相机名字
    camera_names = task_config['camera_names']

    # 索引号
    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir) # 最大为1000
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    
    # 一直循环
    while True: # 计算每个回合 轮   episode翻译成回合
        # 时间间隔, 最大steps默认给1000, 相机名称, 保存路径, 数据名称, 重写
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy:
            break

# 自动索引
def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    
    max_idx = 1000
    # 创建数据集位置
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    
    for i in range(max_idx+1):
        # 如果不是一个文件就返回 i
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

# 打印时间
def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time) 
    
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    
    main(vars(parser.parse_args())) # TODO
    
    # debug()


