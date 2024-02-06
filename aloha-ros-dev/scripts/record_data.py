#!/home/lin/miniconda3/envs/aloha/bin/python

'''
#!/root/miniconda3/envs/aloha/bin/python
#!/home/lin/miniconda3/envs/aloha/bin/python


'''

import os
import time
import numpy as np
# import cv2
import h5py
import tqdm

from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

import rospy

import argparse
import dm_env, collections

# task_name = "aloha_mobile_dummy"
# episode_idx = 2
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', action='store', type=str, help='Task name.', default="aloha_mobile_dummy", required=False)
parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=0, required=False)
parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.', default=500, required=False)
parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.', default="./data", required=False)



# dataset_dir = "/home/lin/test/follow/data"

camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']


topic_img_left ="/camera3/color/image_raw"
topic_img_right="/camera2/color/image_raw"
topic_img_top  ="/camera1/color/image_raw"
topic_master_left = "/master/joint_left"
topic_master_right = "/master/joint_right"
topic_puppet_left = "/puppet/joint_left"
topic_puppet_right = "/puppet/joint_left"


arg = parser.parse_args()
task_name = arg.task_name
episode_idx = arg.episode_idx
max_timesteps = arg.max_timesteps
dataset_dir = arg.dataset_dir

sub_num = 7  # 订阅数  实际订阅7个 不包含最后一维底盘

total_num = 8  # 总共需要发布8个话题 3个图像 4个臂 1个底盘  
sub_flag = np.zeros([total_num]) # 模式，sub_num订阅消息的flag为0， 如果进去了callback就设为true

rospy.init_node('record_episodes', anonymous=True)
bridge = CvBridge()

# 图像类包含左、右、上图像
class Img:
    def __init__(self) -> None:
        self.imgl = None
        self.imgr = None
        self.img_top = None

# 机械臂类 包含动作、位姿、速度、扭矩
class Arm:
    def __init__(self) -> None:
        self.action = None
        self.pose = None
        self.vel = None
        self.effort = None

# 底盘类 包含角速度和线速度
class BaseRobot:
    def __init__(self) -> None:
        self.linear_vel = None
        self.angular_vel = None

img = Img()
master_armL, master_armR, puppet_armL, puppet_armR = [Arm() for _ in range(4)]
base_robot = BaseRobot()



def imageL_callback(msg):
    img.imgl = bridge.imgmsg_to_cv2(msg, 'passthrough')
    sub_flag[0] = True

def imageR_callback(msg):
    img.imgr = bridge.imgmsg_to_cv2(msg, 'passthrough')
    sub_flag[1] = True

def imageTop_callback(msg):
    img.img_top = bridge.imgmsg_to_cv2(msg, 'passthrough')
    sub_flag[2] = True

def masterL_callback(msg):
    master_armL.vel = msg.velocity
    master_armL.pose = msg.position
    master_armL.effort = msg.effort
    master_armL.action = msg.position
    sub_flag[3] = True

def masterR_callback(msg):
    master_armR.vel = msg.velocity
    master_armR.pose = msg.position
    master_armR.effort = msg.effort
    master_armR.action = msg.position
    sub_flag[4] = True

def puppetL_callback(msg):
    puppet_armL.vel = msg.velocity
    puppet_armL.pose = msg.position
    puppet_armL.effort = msg.effort
    puppet_armL.action = msg.position
    sub_flag[5] = True

def puppetR_callback(msg):
    puppet_armR.vel = msg.velocity
    puppet_armR.pose = msg.position
    puppet_armR.effort = msg.effort
    puppet_armR.action = msg.position
    sub_flag[6] = True

def base_robot_callback(msg):
    # msg = Twist()
    base_robot.angular_vel = msg.angular.z
    base_robot.linear_vel = msg.linear.x
    sub_flag[7] = True

# 保存数据函数
def save_data(camera_names, actions, timesteps, max_timesteps, dataset_path):
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
    
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性： 
        # 1 是否仿真
        # 2 图像是否压缩
        # 
        root.attrs['sim'] = False                   
        root.attrs['compress'] = False
        
        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        
        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        _ = obs.create_dataset('qvel', (max_timesteps, 14))
        _ = obs.create_dataset('effort', (max_timesteps, 14))
        _ = root.create_dataset('action', (max_timesteps, 14))
        _ = root.create_dataset('base_action', (max_timesteps, 2))
        
        # data_dict写入h5py.File
        for name, array in data_dict.items():   # 名字+值
            root[name][...] = array

    print(f'Saving: {time.time() - t0:.1f} secs', dataset_path)
    exit(-1)

rospy.Subscriber(topic_img_left, Image, imageL_callback, queue_size=1)
rospy.Subscriber(topic_img_right, Image, imageR_callback, queue_size=1)
rospy.Subscriber(topic_img_top, Image, imageTop_callback, queue_size=1)

rospy.Subscriber(topic_master_left, JointState, masterL_callback, queue_size=1)
rospy.Subscriber(topic_master_right, JointState, masterR_callback, queue_size=1)
rospy.Subscriber(topic_puppet_left, JointState, puppetL_callback, queue_size=1)
rospy.Subscriber(topic_puppet_left, JointState, puppetR_callback, queue_size=1)

# rospy.Subscriber('/base_robot', Twist, base_robot_callback, queue_size=1)


# saving dataset 保存数据路径
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)

dataset_dir = os.path.join(dataset_dir, task_name)

if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)

# 如果文件存在和不需要从重写就退出
if os.path.isfile(dataset_dir):
    print(f'Dataset already exist at \n{dataset_dir}\nHint: set overwrite to True.')
    exit()

dataset_path =  os.path.join(dataset_dir, "episode_"+ str(episode_idx))

timesteps = []
actions = []
# ts = None

# 图像数据
image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
# imgs = ImageRecorder()

image_dict = dict()
for cam_name in camera_names: 
    # setattr(imgs, f'{cam_name}_image', image)
    # image_dict[cam_name] = getattr(imgs, f'{cam_name}_image')
    image_dict[cam_name] = image
# 观察状态


# timesteps.append(ts)

count = 0

input_key = input("please input s:")
while (input_key != 's' and not rospy.is_shutdown()):
    input_key = input("please input s:")


while (count < max_timesteps + 1 and not rospy.is_shutdown()):
    
    # 1 判断全部执行了回调函数
    if not np.all(sub_flag[0:sub_num] == True):
        continue
    
    t1 = time.time()
    count +=1 
    rospy.loginfo(sub_flag)
    sub_flag = np.zeros([total_num])    # 重新设置标志位置
    
    # 2 收集数据  
    # 2.1 图像信息
    image_dict = dict()
    image_dict[camera_names[0]] = img.img_top
    image_dict[camera_names[1]] = img.imgl
    image_dict[camera_names[2]] = img.imgr

    # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
    obs = collections.OrderedDict()  # 有序的字典
    obs['images'] = image_dict
    obs['qpos'] = np.concatenate((np.array(puppet_armL.pose), np.array(puppet_armR.pose)), axis=0)
    obs['qvel'] = np.concatenate((np.array(puppet_armL.vel), np.array(puppet_armR.vel)), axis=0)
    obs['effort'] = np.concatenate((np.array(puppet_armL.effort), np.array(puppet_armR.effort)), axis=0)
    # 2.3 底盘信息
    # obs['base_vel'] = [base_robot.linear_vel, base_robot.angular_vel]
    obs['base_vel'] = [0.0, 0.0]
    
    # 第一帧 只包含first， fisrt只保存StepType.FIRST
    if count == 1:
        ts = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs)
        timesteps.append(ts)
        continue
    
    # 时间步
    ts = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=None,
        discount=None,
        observation=obs)
    
    # 主臂保存状态 
    action = np.concatenate((np.array(master_armL.action), np.array(master_armR.action)), axis=0)
    actions.append(action)
    timesteps.append(ts)
    
    
    print("Frame data: ", count)
    if rospy.is_shutdown():
        exit(-1)
    
    t2 = time.time()
    rospy.sleep(max(0, t2- t1))

print("len(timesteps): ", len(timesteps))
print("len(actions)  : ", len(actions))

if (count < max_timesteps):
    print("failure save_data")
    exit(-1)
save_data(camera_names, actions, timesteps, max_timesteps, dataset_path)

rospy.spin()

# python scripts/record_data.py --max_timesteps 400
