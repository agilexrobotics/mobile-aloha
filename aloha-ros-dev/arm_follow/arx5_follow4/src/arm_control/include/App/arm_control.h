#ifndef _ARM_H_
#define _ARM_H_


#include "utility.h"
#include "Hardware/teleop.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../Hardware/can.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Float64MultiArray.h>
#include "../Hardware/motor.h"
#include <string.h>
//
#include <time.h>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <dirent.h>
#include <vector>
#include <sensor_msgs/JointState.h>

//play
#include "play.h"
#include "solve.h"
#include <termios.h>



// #include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#include <tinyxml2.h>
#include <string>
// #include <iostream>
// #include <iostream>
// #include <string>
// #include <vector>
// #include <filesystem>


//
#define ARX5_PRO_L1 0.27
#define ARX5_PRO_L2 0.2766
#define A4310_kp 160
#define A4310_kD 5
// #define spd 100
// #define cur 1000
#define YAW_WORLD   0
#define BASE  1
#define ELBOW 2
#define PITCH_WAIST 3
#define YAW_WAIST  4
#define ROLL  5
#define GRASP 6

// #define X 0
// #define Y 1
// #define Z 2


#define FORWARD 0
#define DOWNWARD 1

#define JOYSTICK_DEADZONE 0.15 

#define SIM_JOINT_KP 150
#define SIM_JOINT_KD 10

#define filter_sensor 0.3f
#define filter_vel    0.3f
// #define filter_cmd    0.3f
#define filter_torque 0.3f
#define encos_up  2.09f
#define encos_down -2.09f



/* currently, one mission one class, if there exists multi mission and share arm variables, 
either use a more gerenal class including mission instance or use global struct
*/

// struct command
// {
//     float x = 0.0f;
//     float y = 0.0f;
//     float z = 0.0f;
//     float base_yaw = 0.0f;
//     float gripper_roll = 0.0f;
//     float gripper = 0.0f;
//     float waist_yaw = 0.0f;
//     float waist_pitch = 0.0f;
//     bool reset = false;
//     int mode = FORWARD;
// };
struct arx5roscmd
{
    float x=0;
    float y=0;
    float z=0;
    float roll=0;
    float pitch=0;
    float yaw=0;
    float gripper=0;
    int mode1=0;
    int mode2=0;

};


struct cartesian
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct cylinder
{
    float r = 0.0f;
    float phi = 0.0f;
    float z = 0.0f;
};

struct sphere
{
    float rho = 0.0f;
    float phi = 0.0f;
    float theta = 0.0f;
};

struct coordinate
{
    cartesian xyz_pos;
    cylinder cylinder_pos;
    sphere sphere_pos;
};

// enum arx5_state {NORMAL, OUT_RANGE, OUT_BOUNDARY};

class FIFO_Queue
{
public:
    FIFO_Queue(uint queue_size);
    ~FIFO_Queue() = default;

    // void write_data(geometry_msgs::PoseStamped pos_stamped);
    // geometry_msgs::PoseStamped read_nearest_data(ros::Time time);
    uint count = 0;
    // std::vector<geometry_msgs::PoseStamped> data;
    uint write_ptr = 0, read_ptr = 0;
};

class arx_arm
{
public:
    arx_arm(int CONTROL_MODE);
    ~arx_arm()=default;

    bool is_real = false;
    arx5_state command_state;
    int control_mode=0;

    unsigned int loop_rate = 200;

    float current_pos[7] = {};
    float current_vel[7] = {0.0};
    float current_torque[7] = {0.0f};
    float target_pos[7] = {0.0f}, last_target_pos[7] = {0.0f};
    float target_pos_temp[7] = {0.0f};
    float target_vel[7] = {0.0f};
    float ros_control_pos[7] ={};
    float ros_control_pos_t[7] ={};
    float ros_control_vel[7] ={};
    float ros_control_cur[7] ={};
    float slove_cur[7] = {} ;
    float slove_cur_t[7] = {} ;

    // For limiting the orientation of the motors
    float pitch_waist_lb = -1.72;

    float lower_bound_waist[3] = {  0.0, -0.4, -0.4};
    float upper_bound_waist[3] = {  0.4,  0.4,  0.4};

    float lower_bound_pitch = -1.35;
    float upper_bound_pitch = M_PI/2; 
    float lower_bound_yaw = -1.35;
    float upper_bound_yaw = 1.35;

    float max_torque = 15;
    float yaw_base2waist = 0.0f;
    float yaw_world2base = 0.0f;

    float ramp(float final, float now, float ramp_k);
    float joystick_projection(float joy_axis);

    // Get the actual position of joints from CAN controller in real mode
    void get_curr_pos();
    void set_loop_rate(const unsigned int rate);
    // For startup process
    void init_step();
    bool is_starting = true,is_arrived = false, is_init = false;
    int calc_init=0;
    
    float joy_x=0,joy_y=0,joy_z=0,joy_yaw=0,joy_pitch=0,joy_roll=0,joy_gripper=0;
    float joy_x_t=0,joy_y_t=0,joy_z_t=0,joy_yaw_t=0,joy_pitch_t=0,joy_roll_t=0,joy_gripper_t=0;

    std::string model_path; 

    // void kdl_id_calc();
    void teach2pos_mode_step();
    float prev_target_pos[6] = {0};
    bool teach2pos_returning = false;

    float motor_torque_k = 0.72;//0.8  0.75   //0,72

    float kiz = 0.000000001;
    float kipitch = 0.000000001;
    double intergral_z = 0;
    double intergral_pitch = 0;

    KDL::Frame prev_frame;
    void calc_joint_acc();
    float f_x = 0, f_y = 0, f_z = 0, t_r = 0, t_p = 0, t_y = 0;

    ros::NodeHandle nh;

    bool use_random = false;
    const int pause_time = 400; //Step per pause
    void set_rand_joint_pos();
    bool check_valid_rand_pos(KDL::JntArray jointpos);
    void rand_step();
    bool is_rand_moving = false;
    bool is_rand_mode = false;
    bool is_rand_return = false;
    int curr_pause_time = 0;
    bool button5_pressed = false;
    float gripper_rand_pos = 0;
    KDL::JntArray rand_dest_pos = KDL::JntArray(6+1);

    // ros::Publisher joint_pub = nh.advertise<std_msgs::Float64MultiArray>("armcontrol/current_joint_angle", 10);

    // For ROS Control
    bool is_ros_control = false;
    float ros_control_filter = 0.0;
    double k_ros_ctrl = 0.01;
    void ros_ctrl_step();
    // For recording the movements
    bool button3_pressed = 0;
    bool button0_pressed = 0;
    bool button_teach=0;
    bool button_replay=0;

    int  play_flag = 0;

    bool is_torque_control = false;
    bool is_teach_mode = false;


    bool teach_mode = false;

    bool is_recording=false;
    std::string out_teach_path;

    bool read_write_multiple_files = true;
/////////////////////////////////////////////////////////////

    void unify_coordinate();// Old unused function

    command get_cmd(); // ask for teleop cmd
    command arx5_cmd;
    arx5roscmd arx5_ros_cmd;
    float sim_Kp[7] = {SIM_JOINT_KP, SIM_JOINT_KP, SIM_JOINT_KP, SIM_JOINT_KP, SIM_JOINT_KP, SIM_JOINT_KP, SIM_JOINT_KP};
    float sim_Kd[7] = {SIM_JOINT_KD, SIM_JOINT_KD, SIM_JOINT_KD, SIM_JOINT_KD, SIM_JOINT_KD, SIM_JOINT_KD, SIM_JOINT_KD};

    int gimbal_mode = FORWARD;

    bool is_PD_control = true;
    bool is_test_mode = false;


   
    // #real_joint
    int set_sw=0;
    float set_use_kp=200;//200
    float set_use_kd=10;//10
    float set_use_kp15=260;//260
    float set_use_kd15=10;//10
    float set_use_kp26=180;//200
    float set_use_kd26=10;//10


    float tgt_pos[8]         = {(6.357f-0.1f),1.221,-0.734f,-0.744f,  (6.137f+0.1f),4.997f,7.041f,-0.423f};
    float tgt_pos_test[6]    = {0.0f, -60.0, 82.12f, -21.0f, 0.0f, 0.0f};

    float k_imu=0.045f;//0.06
    float total_set_imu=0;
    float Kps[8] = {200.0f, 0.0f, 0.0f, 0.0f,    200.0f, 0.0f, 0.0f, 0.0f};
    float Kds[8] = {10.0f, 15.0f, 10.0f, 10.0f,  10.0f, 15.0f, 10.0f, 10.0f};
    float zeros8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    float lower_bound_sim[7] = { -3.14  ,0      ,-0.1   ,-1.671 ,-1.671,-1.57 ,-9};
    float upper_bound_sim[7] = { 2.618 , 3.14   ,3.24, 1.671 , 1.671, 1.57 ,9};

    float Kp[8] = {A4310_kp,A4310_kp,A4310_kp,A4310_kp,   A4310_kp,A4310_kp,A4310_kp,A4310_kp};
    float Kd[8] = {A4310_kD,A4310_kD,A4310_kD,A4310_kD   ,A4310_kD,A4310_kD,A4310_kD,A4310_kD};
    float ffw_torque[8] = {0.0f};
    float tgt_spds[8] = {0.0f};
    float ffw_torques[8] = {0.0f};
    float body_kp=15.0f,body_kd=1.0f;

    float speed_kpj=1,speed_kij=0.0f,fpj,fij,wheel_speedj,total_fij;
    float v2b_bodycontrol(int safe, int sw2, int ch_1, float imu_pit_gyro, float wheel_speed1, float wheel_speed2);
    float v2b_out = 0;
    void safe_model(void);
    void set_zero(void);
    void init_moto_pos(void);
    void body_balance_UP(void);
    void body_balance(void);
    void update(void);
    void update_real(command cmd);
    void motor_control();
    void joint_control();
    void can_position_control(uint16_t motor_id,float pos,uint16_t spd,uint16_t cur,uint8_t ack_status);
    bool modifyLinkMass(const std::string& inputFilePath, const std::string& outputFilePath, double newMassValue);
    void gripper_control();

    int init_end=0,add_pos=0;
    float set_4005_out=0,set_pos_4005 = 0,k1_4005=400,k2_4005=3,init_pos_4005=0,pos_4005_t=0;
    can CAN_Handlej;
    arx5_play play;
    arx_solve solve;
    std::vector<std::string> play_file_list;
    std::vector<std::string> teach_file_list;
    bool current_normal=true;
    int temp_current_normal=0;
    bool temp_condition=true;
    int temp_init=0;
    float init_kp=0,init_kp_4=0,init_kp_5=0,init_kd=0,init_kd_4=0,init_kd_6=0;

    bool motor_control_cmd= true;
    int rosGetch();
    void getKey(char key_t);
    float gripper_cout=0,gripper_spd,gripper_max_cur=50;
    int restart_flag=0;
    float ros_move_k_x=100.0f,ros_move_k_y=100.0f,ros_move_k_z=100.0f,ros_move_k_yaw=100.0f,ros_move_k_pitch=100.0f,ros_move_k_roll=100.0f;
    void arm_replay_mode();
    //torque_mode
    void arm_torque_mode();
    //reset_mode
    void arm_reset_mode();
    //get_pos
    void arm_get_pos();
    //Recordandreplay teach_mode
    void arm_teach_mode();  
    void limit_pos();   
    float follow_pos_k=1.0f;
    int record_mode=0;
    int teach_press_count = 0;

private:
    // for ros configurations
    // currently not necessary
    

    ros::Publisher joint_state_pub;
    sensor_msgs::JointState joint_state;

    ros::Publisher joint_cmd_pub;
    sensor_msgs::JointState joint_cmd;

    float test_pos = 0;
    uint test_cnt = 0;

    LowPassFilter *z_fd_filter;
    LowPassFilter *joint_vel_filter[8];
    LowPassFilter *target_vel_filter[8];

   
};

#endif