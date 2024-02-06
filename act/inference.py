#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange
from constants import DT
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
from threading import Thread

import sys
sys.path.append("./")

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


def get_model_config(args):
    # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
    set_seed(1)
    # get task parameters # 是否是仿真数据
    is_sim = args.task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[args.task_name]
    # 不是仿真数据就加载aloha_scripts.constants的配置文件
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[args.task_name]

    # 如果是ACT策略
    if args.policy_class == 'ACT':
        policy_config = {'lr': args.lr,
                         'num_queries': args.chunk_size,     # 查询
                         'kl_weight': args.kl_weight,        # kl散度权重
                         'hidden_dim': args.hidden_dim,      # 隐藏层维度
                         'dim_feedforward': args.dim_feedforward,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'enc_layers': args.enc_layers,
                         'dec_layers': args.dec_layers,
                         'nheads': args.nheads,
                         'camera_names': task_config['camera_names'],
                         }
    # 如果cnn, num_queries': 1
    elif args.policy_class == 'CNNMLP':
        policy_config = {'lr': args.lr, 'lr_backbone': args.lr_backbone, 'backbone': args.backbone, 'num_queries': 1,
                         'camera_names': task_config['camera_names'],}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': args.num_epochs,
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'ckpt_stats_name': args.ckpt_stats_name,
        'episode_len': args.max_publish_steps,
        'state_dim': args.state_dim,
        'lr': args.lr,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'task_name': args.task_name,
        'seed': args.seed,
        'temporal_agg': args.temporal_agg,
        'camera_names': task_config['camera_names'],
    }
    return config


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)  
    
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
    
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def inference_process(args, config, ros_operator, policy, stats, t):
    global inference_lock
    global inference_actions
    global inference_timestep
    print_flag = True
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, robot_base) = result
        image_dict = dict()
        image_dict[config['camera_names'][0]] = img_front
        image_dict[config['camera_names'][1]] = img_left
        image_dict[config['camera_names'][2]] = img_right
        obs = collections.OrderedDict()
        obs['images'] = image_dict
        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate(
            (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate(
            (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        # obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
        obs['base_vel'] = [0.0, 0.0]

        # 取obs的位姿qpos
        qpos_numpy = np.array(obs['qpos'])

        # 归一化处理qpos 并转到cuda
        qpos = pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        # 当前图像curr_image获取图像
        curr_image = get_image(obs, config['camera_names'])
        all_actions = policy(qpos, curr_image)
        all_actions_inter = np.zeros([1, (config['policy_config']['num_queries'] - 1) * args.inference_rate_scale + 1, config['state_dim']])
        for i in range(len(all_actions_inter[0])):
            if i % args.inference_rate_scale == 0:
                all_actions_inter[0][i] = all_actions[0][int(i / args.inference_rate_scale)].cpu().detach().numpy()
            else:
                front = int(i / args.inference_rate_scale)
                tail = front + 1
                for j in range(len(all_actions[0][front])):
                    all_actions_inter[0][i][j] = (all_actions[0][front][j] + (
                                all_actions[0][tail][j] - all_actions[0][front][j]) * (
                                                             (i % args.inference_rate_scale) / args.inference_rate_scale)).cpu().detach().numpy()
        inference_lock.acquire()
        inference_actions = all_actions_inter
        inference_timestep = t
        inference_lock.release()
        break


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)

    # 1 创建模型数据  继承nn.Module
    policy = make_policy(config['policy_class'], config['policy_config'])
    # print("model structure\n", policy.model)
    
    # 2 加载模型权重
    ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    if not loading_status:
        print("ckpt path not exist")
        return False

    # print(loading_status)
    # from torchsummary import summary
    # qpos: batch, qpos_dim     [1,14]
    # image: batch, num_cam, channel, height, width [1,3, 3 480, 640]
    # env_state: None                                   没有值None
    # actions: batch, seq, action_dim       [B, 14, 512]
    # summary(policy, [(1, 14), (1, 3, 480, 640),(0),(1, 100, 14)])

    # 3 模型设置为cuda模式和验证模式
    policy.cuda()
    policy.eval()

    # 4 加载统计值
    stats_path = os.path.join(config['ckpt_dir'], config['ckpt_stats_name'])
    # 统计的数据  # 加载action_mean, action_std, qpos_mean, qpos_std 14维
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 数据预处理和后处理函数定义
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    max_publish_steps = config['episode_len']
    inference_steps = (config['policy_config']['num_queries'] - 1) * args.inference_rate_scale + 1
    max_inference_steps = (config['policy_config']['num_queries'] - 1) * args.inference_rate_scale + 1
    num_queries = config['policy_config']['num_queries']
    if config['temporal_agg']:
        inference_steps = 1 * args.inference_rate_scale

    # 发布基础的姿态
    left = [-0.00133514404296875, 0.00286102294921875, 0.01621246337890625, -0.02574920654296875, -0.00095367431640625, -0.00362396240234375, 1.179330825805664]
    right = [-0.00057220458984375, -0.00019073486328125, 0.01850128173828125, 0.00743865966796875, 0.00133514404296875, 0.00019073486328125, -0.01544952392578125]
    ros_operator.puppet_arm_publish_continuous(left, right)

    # 推理
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # 每个回合的步数
            rate = rospy.Rate(args.publish_rate)
            print_flag = True
            t = 0
            max_t = 0
            all_time_actions = None
            if config['temporal_agg']:
                all_time_actions = np.zeros([max_publish_steps, max_publish_steps + num_queries * args.inference_rate_scale, config['state_dim']])
            while t < max_publish_steps and not rospy.is_shutdown():
                start_time = time.time()
                # query policy
                if config['policy_class'] == "ACT":
                    if t % inference_steps == 0:
                        if inference_thread is None:
                            inference_thread = threading.Thread(target=inference_process, args=(args, config, ros_operator, policy, stats, t))
                            inference_thread.start()
                            print(t)
                    if inference_thread is not None:  #  and t >= max_t:
                        inference_thread.join()
                    inference_lock.acquire()
                    if inference_actions is not None:
                        inference_thread.join()
                        inference_thread = None
                        all_actions = inference_actions
                        inference_actions = None
                        if config['temporal_agg']:
                            all_time_actions[[inference_timestep], inference_timestep:inference_timestep + max_inference_steps] = all_actions
                        max_t = inference_timestep + max_inference_steps
                    inference_lock.release()

                    # if t % inference_steps == 0:  # 取余
                    #     result = ros_operator.get_frame()
                    #     if not result:
                    #         if print_flag:
                    #             print("syn fail")
                    #             print_flag = False
                    #         rate.sleep()
                    #         continue
                    #     print_flag = True
                    #     (img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, robot_base) = result
                    #     image_dict = dict()
                    #     image_dict[config['camera_names'][0]] = img_front
                    #     image_dict[config['camera_names'][1]] = img_left
                    #     image_dict[config['camera_names'][2]] = img_right
                    #     obs = collections.OrderedDict()
                    #     obs['images'] = image_dict
                    #     obs['qpos'] = np.concatenate(
                    #         (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
                    #     obs['qvel'] = np.concatenate(
                    #         (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
                    #     obs['effort'] = np.concatenate(
                    #         (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
                    #     # obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
                    #     obs['base_vel'] = [0.0, 0.0]
                    #
                    #     # 取obs的位姿qpos
                    #     qpos_numpy = np.array(obs['qpos'])
                    #
                    #     # 归一化处理qpos 并转到cuda
                    #     qpos = pre_process(qpos_numpy)
                    #     qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    #
                    #     # 当前图像curr_image获取图像
                    #     curr_image = get_image(obs, config['camera_names'])
                    #     all_actions = policy(qpos, curr_image)
                    #     all_actions_inter = torch.zeros([1, (num_queries - 1) * args.inference_rate_scale + 1, config['state_dim']]).cuda()
                    #     for i in range(len(all_actions_inter[0])):
                    #         if i % args.inference_rate_scale == 0:
                    #             all_actions_inter[0][i] = all_actions[0][int(i/args.inference_rate_scale)]
                    #         else:
                    #             front = int(i/args.inference_rate_scale)
                    #             tail = front + 1
                    #             for j in range(len(all_actions[0][front])):
                    #                 all_actions_inter[0][i][j] = all_actions[0][front][j] + (all_actions[0][tail][j] - all_actions[0][front][j]) * ((i % args.inference_rate_scale) / args.inference_rate_scale)
                    #     all_actions = all_actions_inter
                    #     if config['temporal_agg']:
                    #         all_time_actions[[t], t:t + ((num_queries - 1) * args.inference_rate_scale + 1)] = all_actions
                    if config['temporal_agg']:
                        # raw_action = all_actions[:, t % inference_steps]
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        print(actions_for_curr_step.shape)
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        # exp_weights = exp_weights.unsqueeze(dim=1)
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                        # raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        # 推出 t % inference_steps
                        raw_action = all_actions[:, t % inference_steps]
                # elif config['policy_class'] == "CNNMLP":
                #     raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                # 后处理到raw_action
                # 转到cpu 归一化
                # raw_action = raw_action.squeeze(0).cpu().numpy()
                # action = post_process(raw_action)   # 反归一化处理 均值和方差
                action = post_process(raw_action[0])
                left_action = action[:7]  # 取7维度
                right_action = action[7:]
                ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
                t += 1
                end_time = time.time()
                elapsed_time = end_time - start_time
                # print("time：", elapsed_time)
                print("publish: ", t)
                print(left_action)
                print(right_action)
                rate.sleep()
    ros_operator.puppet_arm_publish_continuous(left, right)


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.bridge = None
        self.img_left_deque = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0:
            return False
        frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(),
                          self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, robot_base

    def img_left_callback(self, msg):
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        self.img_front_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--max_publish_steps', action='store', type=int, help='max_publish_steps', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=30, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera2/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera1/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera3/color/image_raw', required=False)
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--arm_error', action='store', type=float, help='arm_error',
                        default=0.01, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=30, required=False)
    parser.add_argument('--inference_rate_scale', action='store', type=int, help='inference_rate_scale',
                        default=1, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.0], required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_model_config(args)
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()
