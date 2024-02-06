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
float calc_cur[7]={};

int  touch_body = 0;

int  touch_body_t[7] = {};
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
    ros::init(argc, argv, "arm_4"); 
    ros::NodeHandle node;
    Teleop_Use()->teleop_init(node);

    arx_arm ARX_ARM((int) CONTROL_MODE);


    // ros::Subscriber sub_joint = node.subscribe<arm_control::JointControl>("joint_control", 10, 
    //                               [&ARX_ARM](const arm_control::JointControl::ConstPtr& msg) {
    //                                 //   ARX_ARM.ros_control_pos_t[0] = msg->joint_pos[0];
    //                                 //   ARX_ARM.ros_control_pos_t[1] = msg->joint_pos[1];
    //                                 //   ARX_ARM.ros_control_pos_t[2] = msg->joint_pos[2];
    //                                 //   ARX_ARM.ros_control_pos_t[3] = msg->joint_pos[3];
    //                                 //   ARX_ARM.ros_control_pos_t[4] = msg->joint_pos[4];
    //                                 //   ARX_ARM.ros_control_pos_t[5] = msg->joint_pos[5];
    //                                 //   ARX_ARM.ros_control_pos_t[6] = msg->joint_pos[6];
    //                                   ARX_ARM.record_mode = msg->mode;
    //                                 //   ROS_WARN("debug>>>>>");
    //                               });

   
    ros::Subscriber sub_joint = node.subscribe<sensor_msgs::JointState>("/master/joint_left", 10, 
                                [&ARX_ARM](const sensor_msgs::JointState::ConstPtr& msg) 
                                {
                                    ARX_ARM.ros_control_pos_t[0] = msg->position[0];
                                    ARX_ARM.ros_control_pos_t[1] = msg->position[1];
                                    ARX_ARM.ros_control_pos_t[2] = msg->position[2];
                                    ARX_ARM.ros_control_pos_t[3] = msg->position[3];
                                    ARX_ARM.ros_control_pos_t[4] = msg->position[4];
                                    ARX_ARM.ros_control_pos_t[5] = msg->position[5];
                                    ARX_ARM.ros_control_pos_t[6] = msg->position[6];
                                    ARX_ARM.record_mode = 0;
                                });    




    // 发布
    // ros::Publisher pub = node.advertise<arm_control::arx5>("arx5cmd", 10);
    // ros::Publisher pub_chassis = node.advertise<arm_control::ChassisCtrl>("/chassis_ctrl", 10);
    // ros::Publisher pub_current = node.advertise<arm_control::JointInformation>("joint_information", 10);
    ros::Publisher pub_joint01 = node.advertise<sensor_msgs::JointState>("/puppet/joint_left", 10);

    arx5_keyboard ARX_KEYBOARD;

    ros::Rate loop_rate(200);
    ARX_ARM.set_loop_rate(200);
    can CAN_Handlej;

    std::thread keyThread(&arx5_keyboard::detectKeyPress, &ARX_KEYBOARD);
    sleep(1);

    while(ros::ok())
    { 

        ROS_INFO("follow2>>>>>>>>>>>>>>>>>>>>");
        //                                                                ARX_ARM.arx5_ros_cmd.roll,ARX_ARM.arx5_ros_cmd.pitch,ARX_ARM.arx5_ros_cmd.yaw );

        char key = ARX_KEYBOARD.keyPress.load();
        ARX_ARM.getKey(key);

        ARX_ARM.get_curr_pos();
        if(!ARX_ARM.is_starting){
             cmd = ARX_ARM.get_cmd();
        }
        ARX_ARM.update_real(cmd);



        arm_control::JointInformation msg_joint;       

        for(int i=0;i<6;i++)
        {
            calc_cur[i]=ARX_ARM.current_torque[i]-ARX_ARM.slove_cur[i];
        }


         for(int i=0;i<6;i++)
        {    
            if(abs(calc_cur[i]) >3)
            {
                touch_body_t[i]++;
            }else{
                touch_body_t[i]=0;
            }
        }



         for(int i=0;i<6;i++)
        { 
            if(touch_body_t[i]>10) //
            msg_joint.joint_cur[i]=-ARX_ARM.current_torque[i]*0.3;
            else{
                msg_joint.joint_cur[i]=0;
            }
        }

        msg_joint.joint_cur[6]=-ARX_ARM.current_torque[6]*0.1f;  //0.3
        



        // pub_current.publish(msg_joint);
        touch_body =0;

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
        }
        pub_joint01.publish(msg_joint01);


        ros::spinOnce();
        loop_rate.sleep();
        
    }
    // keyThread.join(); 
    return 0;
}