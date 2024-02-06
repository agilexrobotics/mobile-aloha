#!/home/lin/software/miniconda3/envs/aloha/bin/python

'''
    #!/usr/bin/env python

'''
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist

import cv2
import numpy as np




rospy.init_node('joint_state_publisher', anonymous=True)

bridge = CvBridge()

# 创建一个发布者，发布到名为 '/joint_states' 的话题上，消息类型为 JointState
joint_state_publisher1 = rospy.Publisher('/joint_states1', JointState, queue_size=10)
joint_state_publisher2 = rospy.Publisher('/joint_states2', JointState, queue_size=10)


image_publisher1 = rospy.Publisher("/image_left", Image, queue_size=10)
image_publisher2 = rospy.Publisher("/image_right", Image, queue_size=10)
image_publisher3 = rospy.Publisher("/image_top", Image, queue_size=10)

base_publisher = rospy.Publisher("/base_robot", Twist, queue_size=10)

rate = rospy.Rate(10) 


def publish_joint_state():
    # 创建一个 JointState 消息
    joint_state_msg = JointState()
    joint_state_msg.header = Header()
    joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
    
    joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper']  # 设置关节名称
    
    random_position = np.random.rand(7)  # 生成一个在[0.0, 1.0)范围内的随机浮点数
    print(random_position)

    # joint_state_msg.position = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 0.1]  # 设置关节位置
    # joint_state_msg.velocity = [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.01]  # 设置关节速度
    # joint_state_msg.effort = [10.0, 20.0, 30.0,10.0, 20.0, 30.0,10]  # 设置关节力矩

    joint_state_msg.position = random_position
    joint_state_msg.velocity = random_position
    joint_state_msg.effort   = random_position

    # image = np.ones((480, 640, 3), np.uint8) * 127

    image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    img_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")


    base_robot_msg = Twist()
    base_robot_msg.angular.z = 0.1
    base_robot_msg.linear.x = 0.2
    print(base_robot_msg)

    # 发布关节状态信息
    joint_state_publisher1.publish(joint_state_msg)
    joint_state_publisher2.publish(joint_state_msg)
    image_publisher1.publish(img_msg)
    image_publisher2.publish(img_msg)
    image_publisher3.publish(img_msg)
    base_publisher.publish(base_robot_msg)

    rospy.loginfo("pub action successfully")

    rate.sleep()


if __name__ == '__main__':
    # 初始化节点
    while not rospy.is_shutdown():
        publish_joint_state()
    rospy.spin()