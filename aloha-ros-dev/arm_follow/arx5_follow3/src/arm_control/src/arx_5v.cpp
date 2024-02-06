#include <ros/ros.h>
#include <cmath>
#include <iostream>
#include <std_msgs/Float32MultiArray.h>
#include "utility.h"
#include "Hardware/can.h"
#include "Hardware/motor.h"
#include "Hardware/teleop.h"
#include "App/arm_control.h"
#include "App/arm_control.cpp"
#include "App/keyboard.h"
#include "App/play.h"
#include "App/solve.h"
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>
#include <atomic>
#include "arm_control/arx5.h"
#include "arm_control/JointControl.h"
#include "arm_control/JointInformation.h"
#include "arm_control/ChassisCtrl.h"
#include "arm_control/MagicCmd.h"

#include <sensor_msgs/JointState.h>

#define IS_REAL 0

int CONTROL_MODE=0; // 0 arx5 rc ，1 5a rc ，2 arx5 ros ，3 5a ros
command cmd;
float chassis_vx,chassis_vy,chassis_vz;
void magicCallback(const arm_control::MagicCmd::ConstPtr& msg)
{
    
    magic_pos[0] = msg->x;
    magic_pos[1] = msg->y;
    magic_pos[2] = msg->z;
    magic_angle[0] = msg->pitch;
    magic_angle[1] = msg->yaw;
    magic_angle[2] = msg->roll;

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_3"); 
    ros::NodeHandle node;
    Teleop_Use()->teleop_init(node);

    arx_arm ARX_ARM((int) CONTROL_MODE);

    // 订阅
    // ros::Subscriber sub = node.subscribe("arx5cmd", 10, myCallback);

    // sub_magic = node.subscribe<arm_control::MagicCmd>("/magic_cmd", 10, 
    //                               [&ARX_ARM](const arm_control::MagicCmd::ConstPtr& msg) {
    //                                   watch_cmd[0] = msg->x;
    //                                   watch_cmd[1] = msg->y;
    //                                   watch_cmd[2] = msg->z;
    //                                   watch_cmd[3] = msg->pitch;
    //                                   watch_cmd[4] = msg->yaw;
    //                                   watch_cmd[5] = msg->roll;
    //                                 //   magic_pos[0] = msg->x;
    //                                 //   magic_pos[1] = msg->y;
    //                                 //   magic_pos[2] = msg->z;
    //                                 //   magic_angle[0] = msg->pitch;
    //                                 //   magic_angle[1] = msg->yaw;
    //                                 //   magic_angle[2] = msg->roll;
        
    //                               });

    // ros::Subscriber sub_magic = node.subscribe<arm_control::MagicCmd>("magic_cmd", 10, magicCallback);

    // ros::Subscriber sub_joint = node.subscribe<arm_control::JointControl>("joint_control", 10, 
    //                               [&ARX_ARM](const arm_control::JointControl::ConstPtr& msg) {
    //                                   ARX_ARM.ros_control_pos_t[0] = msg->joint_pos[0];
    //                                   ARX_ARM.ros_control_pos_t[1] = msg->joint_pos[1];
    //                                   ARX_ARM.ros_control_pos_t[2] = msg->joint_pos[2];
    //                                   ARX_ARM.ros_control_pos_t[3] = msg->joint_pos[3];
    //                                   ARX_ARM.ros_control_pos_t[4] = msg->joint_pos[4];
    //                                   ARX_ARM.ros_control_pos_t[5] = msg->joint_pos[5];
    //                                 //   ROS_WARN("debug>>>>>");
    //                               });


    ros::Subscriber sub_information = node.subscribe<arm_control::JointInformation>("/joint_informationL", 10, 
                                  [&ARX_ARM](const arm_control::JointInformation::ConstPtr& msg) {
                                      ARX_ARM.ros_control_cur_t[0] = msg->joint_cur[0];
                                      ARX_ARM.ros_control_cur_t[1] = msg->joint_cur[1];
                                      ARX_ARM.ros_control_cur_t[2] = msg->joint_cur[2];
                                      ARX_ARM.ros_control_cur_t[3] = msg->joint_cur[3];
                                      ARX_ARM.ros_control_cur_t[4] = msg->joint_cur[4];
                                      ARX_ARM.ros_control_cur_t[5] = msg->joint_cur[5];
                                      ARX_ARM.ros_control_cur_t[6] = msg->joint_cur[6];
                                    //   ROS_WARN("debug>>>>>");
                                  });



    // 发布
    ros::Publisher pub_joint01 = node.advertise<sensor_msgs::JointState>("/master/joint_left", 10);
    

    arx5_keyboard ARX_KEYBOARD;

    ros::Rate loop_rate(200);
    ARX_ARM.set_loop_rate(200);
    can CAN_Handlej;

    std::thread keyThread(&arx5_keyboard::detectKeyPress, &ARX_KEYBOARD);
    sleep(1);

    while(ros::ok())
    { 

        ROS_INFO("master 2>>>>>>>>>>>>>>>>>>>>>");
        //                                                                ARX_ARM.arx5_ros_cmd.roll,ARX_ARM.arx5_ros_cmd.pitch,ARX_ARM.arx5_ros_cmd.yaw );
        char key = ARX_KEYBOARD.keyPress.load();
        ARX_ARM.getKey(key);

        ARX_ARM.get_curr_pos();
        if(!ARX_ARM.is_starting){
             cmd = ARX_ARM.get_cmd();
        }
        ARX_ARM.update_real(cmd);
    
      
            //发布关节信息
            // arm_control::JointControl msg_joint;            
            // msg_joint.joint_pos[0]=ARX_ARM.current_pos[0];
            // msg_joint.joint_pos[1]=ARX_ARM.current_pos[1];
            // msg_joint.joint_pos[2]=ARX_ARM.current_pos[2];
            // msg_joint.joint_pos[3]=ARX_ARM.current_pos[3];
            // msg_joint.joint_pos[4]=ARX_ARM.current_pos[4];
            // msg_joint.joint_pos[5]=ARX_ARM.current_pos[5];
            // msg_joint.joint_pos[6]=ARX_ARM.current_pos[6]*9;  //10
            
            // pub_joint.publish(msg_joint);

            ROS_INFO("master_joint_left");
            sensor_msgs::JointState msg_joint01;
            msg_joint01.header.stamp = ros::Time::now();
            // msg_joint01.header.frame_id = "map";
            size_t num_joint = 7;
            msg_joint01.name.resize(num_joint);
            msg_joint01.velocity.resize(num_joint);
            msg_joint01.position.resize(num_joint);
            msg_joint01.effort.resize(num_joint);
            
            for (size_t i=0; i < 7; ++i)
            {   
                msg_joint01.name[i] = "joint" + std::to_string(i);
                msg_joint01.position[i] = ARX_ARM.current_pos[i];
                msg_joint01.velocity[i] = ARX_ARM.current_vel[i];
                msg_joint01.effort[i] = ARX_ARM.current_torque[i];
                if (i == 6) msg_joint01.position[i] *=9;
            }
            pub_joint01.publish(msg_joint01);




            // if(ARX_ARM.arx5_cmd.key_u)
            // {
            //     chassis_vx = 1;
            // }else if(ARX_ARM.arx5_cmd.key_j)
            // {
            //     chassis_vx = -1;
            // }else chassis_vx=0;


            // if(ARX_ARM.arx5_cmd.key_h)
            // {
            //     chassis_vz = 1;
            // }else if(ARX_ARM.arx5_cmd.key_k)
            // {
            //     chassis_vz = -1;
            // }else chassis_vz=0;

            // arm_control::ChassisCtrl msg_chassis;            
            // msg_chassis.vx=chassis_vx;
            // msg_chassis.vz=chassis_vz;
            // msg_chassis.mode1 = (int)ARX_ARM.arx5_cmd.key_v;
            // msg_chassis.mode2 = (int)ARX_ARM.arx5_cmd.key_b;
            // pub_chassis.publish(msg_chassis);

        ros::spinOnce();
        loop_rate.sleep();
        
    }
    // keyThread.join(); 
    return 0;
}