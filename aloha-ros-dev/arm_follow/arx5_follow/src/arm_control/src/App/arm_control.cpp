#include "App/arm_control.h"


#define postion_control_spd 300
#define postion_control_cur 1000
extern OD_Motor_Msg rv_motor_msg[10];
extern m_rmd_t rmd_01;
extern float magic_pos[3];
extern float magic_angle[3];

float arx_arm::joystick_projection(float joy_axis)
/**
 * @brief joystick dead band
 */
{
    float cmd = 0.0f;
    if(abs(joy_axis) >= JOYSTICK_DEADZONE)
    {
        cmd = (joy_axis - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE);
    }
    return cmd;
}

float arx_arm::ramp(float goal, float current, float ramp_k)
{
    float retval = 0.0f;
    float delta = 0.0f;
    delta = goal - current;
    if (delta > 0)
    {
        if (delta > ramp_k)
        {  
                current += ramp_k;
        }   
        else
        {
                current += delta;
        }
    }
    else
    {
        if (delta < -ramp_k)
        {
                current += -ramp_k;
        }
        else
        {
                current += delta;
        }
    }	
    retval = current;
    return retval;
}

arx_arm::arx_arm(int CONTROL_MODE)
{


    // is_real = real_flg;
    control_mode=CONTROL_MODE;

    arx5_cmd.reset = false;
    arx5_cmd.x = 0.0;
    arx5_cmd.y = 0.0;
    arx5_cmd.z = 0.01; // if 0.0 is also ok but isaac sim joint 3 direction will be confused
    arx5_cmd.base_yaw = 0;
    arx5_cmd.gripper = 0;
    arx5_cmd.gripper_roll = 0;
    arx5_cmd.waist_pitch = 0;
    arx5_cmd.waist_yaw = 0;
    arx5_cmd.mode = FORWARD;

    // Read robot model from urdf model
    if(control_mode == 0 ||  control_mode == 2){
        model_path = ros::package::getPath("arm_control") + "/models/ultron_v1.1_aa.urdf"; 
    }else if(control_mode == 1 ||  control_mode == 3) {
        model_path = ros::package::getPath("arm_control") + "/models/arx5h.urdf";
    }
    
    if (modifyLinkMass(model_path, model_path, 0.650)) { //默认单位kg 夹爪质量 0.381 0.581
        std::cout << "Successfully modified the mass of link6." << std::endl;
    } else {
        std::cout << "Failed to modify the mass of link6." << std::endl;
    }    
    
    solve.solve_init(model_path);

    z_fd_filter = new LowPassFilter(200, 10);
    joint_vel_filter[0] = new LowPassFilter(200, 20);
    joint_vel_filter[1] = new LowPassFilter(200, 10);
    joint_vel_filter[2] = new LowPassFilter(200, 20);
    joint_vel_filter[3] = new LowPassFilter(200, 20);
    joint_vel_filter[4] = new LowPassFilter(200, 10);
    joint_vel_filter[5] = new LowPassFilter(200, 20);
    target_vel_filter[4] = new LowPassFilter(200, 30);


    if(control_mode == 0 || control_mode == 2){
    CAN_Handlej.Can_cmd_all(1, 0, 0, 0, 0, 0);        
    }else{CAN_Handlej.Enable_Moto(0x01); }
    usleep(1000);
    CAN_Handlej.Can_cmd_all(2, 0, 0, 0, 0, 0);
    usleep(1000);
    CAN_Handlej.Can_cmd_all(4, 0, 0, 0, 0, 0);
    usleep(1000);
    CAN_Handlej.Enable_Moto(0x05);
    usleep(1000);
    CAN_Handlej.Enable_Moto(0x06);
    usleep(1000);
    CAN_Handlej.Enable_Moto(0x07);
    usleep(1000);
    CAN_Handlej.Enable_Moto(0x08);
    usleep(1000);   

}

void arx_arm::get_curr_pos()
{

    // current_pos[0] = rv_motor_msg[0].angle_actual_rad * (1 - filter_sensor) + current_pos[0] * filter_sensor;
    // current_pos[1] = rv_motor_msg[1].angle_actual_rad * (1 - filter_sensor) + current_pos[1] * filter_sensor;
    // current_pos[2] = rv_motor_msg[3].angle_actual_rad * (1 - filter_sensor) + current_pos[2] * filter_sensor;

    current_pos[0] = rv_motor_msg[0].angle_actual_rad;
    current_pos[1] = rv_motor_msg[1].angle_actual_rad;
    current_pos[2] = rv_motor_msg[3].angle_actual_rad;
    current_pos[3] = rv_motor_msg[4].angle_actual_rad;
    current_pos[4] = rv_motor_msg[5].angle_actual_rad;
    current_pos[5] = rv_motor_msg[6].angle_actual_rad;
    current_pos[6] = rv_motor_msg[7].angle_actual_rad;
    // current_pos[6] = rv_motor_msg[8].gripper_totalangle;

    // ROS_ERROR("current_pos %f ",current_pos[6]);
    for(int i=0;i<5;++i)
    {
        if(current_pos[i]==0)
        ROS_ERROR("motor %d is not connected",i+1);
    }

    current_torque[0] = rv_motor_msg[0].current_actual_float;
    current_torque[1] = rv_motor_msg[1].current_actual_float;
    current_torque[2] = rv_motor_msg[3].current_actual_float;
    current_torque[3] = rv_motor_msg[4].current_actual_float;
    current_torque[4] = rv_motor_msg[5].current_actual_float;
    current_torque[5] = rv_motor_msg[6].current_actual_float;

    int set_max_torque = 9;
    
        for (float num : current_torque) {
            if (abs(num) > set_max_torque) {
                temp_current_normal++;
            }
        }

        for (float num : current_torque) {
            if(abs(num) > set_max_torque)
            {
                temp_condition = false;
            }
        }
        if(temp_condition)
        {
            temp_current_normal=0;
        }

        if(temp_current_normal>300)
        {
            current_normal = false;
        }

}

void arx_arm::set_loop_rate(const unsigned int rate)
{
    loop_rate = rate;
    return;
}

command arx_arm::get_cmd()
{
   
    // replay_mode();
    arm_replay_mode();

    if (is_recording)
    {
        play.update_record(current_pos);
        printf("is currently recording \n");
    }

    if (play.is_playing)
    {
        return arx5_cmd;
    }
    else
    {
        arm_torque_mode();
        arm_reset_mode();
        arm_get_pos();
        //Recordandreplay teach_mode
        // arm_teach_mode();                                      
    }
    return arx5_cmd;
}

void arx_arm::update_real(command cmd)
{

    gimbal_mode = cmd.mode;

    // Move gimbal by inverse calculation or random joint positions
    arx5_state state = NORMAL;

    if(is_starting)
    {
        motor_control_cmd=false;
        init_step();
    }
    else
    {
        if(control_mode == 0  ||  control_mode == 1)//解算
        {
            if(play.is_playing)
            {
                    play_flag=0;
                    is_starting=0;
                    temp_init=0;
                    motor_control_cmd=false;
                    play.replay_step(target_pos,current_pos,play_file_list); 
                    ROS_INFO("replay_step\n");
            }
            else
            {
                state = solve.kdl_ik_calc(arx5_cmd,current_pos,target_pos,is_teach_mode);
            }

                if(state == OUT_BOUNDARY) ROS_ERROR("waist command out of joint boundary!");
                else if(state == OUT_RANGE) 
                {
                    // ROS_ERROR("waist command out of arm range!");
                }
                //position or torque control
                if(teach2pos_returning ){
                    motor_control_cmd=false;
                    is_starting=1;
                    arx5_cmd.reset = true;
                    arx5_cmd.waist_pitch  = arx5_cmd.waist_pitch_t  =arx5_cmd.control_pit   = joy_pitch_t  =joy_pitch      =0      ;
                    arx5_cmd.x            = arx5_cmd.x_t            =arx5_cmd.control_x   = joy_x_t      =joy_x            =0.0 ;
                    arx5_cmd.y            = arx5_cmd.y_t            =arx5_cmd.control_y  = joy_y_t      =joy_y             =0.0  ;
                    arx5_cmd.z            = arx5_cmd.z_t            =arx5_cmd.control_z   = joy_z_t      =joy_z            =0.0  ;      
                    arx5_cmd.base_yaw     = arx5_cmd.base_yaw_t     =      0 ;
                    arx5_cmd.gripper_roll = arx5_cmd.gripper_roll_t =arx5_cmd.control_roll   = joy_roll_t   =joy_roll       =0      ;
                    arx5_cmd.waist_yaw    = arx5_cmd.waist_yaw_t    =arx5_cmd.control_yaw   = joy_yaw_t    =joy_yaw        =0      ;
                    arx5_cmd.mode = FORWARD;
                }
                else{
                    is_starting=0;
                    temp_init=0;
                    motor_control();

                    // gripper_control();
                }
        }

        if(control_mode == 2  ||  control_mode == 3)//直接控制关节
        {
            joint_control();
            // ROS_WARN("ROS_control>>>>>>>>>");
        }

    }

    return;
}

void arx_arm::motor_control()
{
    float pos_ramp_k=10;
    target_pos_temp[0]=ramp( target_pos[0], target_pos_temp[0], pos_ramp_k);    
    target_pos_temp[1]=ramp( target_pos[1], target_pos_temp[1], pos_ramp_k);
    target_pos_temp[2]=ramp( target_pos[2], target_pos_temp[2], pos_ramp_k);
    target_pos_temp[3]=ramp( target_pos[3], target_pos_temp[3], pos_ramp_k);
    target_pos_temp[4]=ramp( target_pos[4], target_pos_temp[4], pos_ramp_k);
    target_pos_temp[5]=ramp( target_pos[5], target_pos_temp[5], pos_ramp_k);


    motor_control_cmd=true;

    cur_change();


    if(current_normal)
    {
        //     if (is_teach_mode) {
        //         if(control_mode == 0 || control_mode==2){
        //             CAN_Handlej.Can_cmd_all (1, 0, 0, 0, 0,  solve.jointTorques(0)+ros_control_cur_t[0]);
        //         }else if(control_mode == 1 || control_mode==3){
        //             CAN_Handlej.Send_DM_Moto(1, 0, 0, 0, 0,  solve.jointTorques(0)+ros_control_cur_t[0]);
        //         }
        //         usleep(200);
        // /////////////////////////////torque/////////////////////////////////////
        //         CAN_Handlej.Can_cmd_all (2, 0, 0, 0, 0,  solve.jointTorques(1)+ros_control_cur_t[1]);
        //         usleep(200);
        //         CAN_Handlej.Can_cmd_all (4, 0, 0, 0, 0,  solve.jointTorques(2)+ros_control_cur_t[2]);
        //         usleep(200);
        //         CAN_Handlej.Send_DM_Moto(5, 0, 0, 0, 0,  solve.jointTorques(3)+ros_control_cur_t[3]);
        //         usleep(200);
        //         CAN_Handlej.Send_DM_Moto(6, 0, 0, 0, 0,  solve.jointTorques(4)+ros_control_cur_t[4]);
        //         usleep(200);
        //         CAN_Handlej.Send_DM_Moto(7, 0, 0, 0, 0,  solve.jointTorques(5)+ros_control_cur_t[5]);
        //         usleep(200);
        float teach_k=1.5f;
            if (is_teach_mode) {
                if(control_mode == 0 || control_mode==2){
                    CAN_Handlej.Can_cmd_all (1, 0, 0, 0, 0,  solve.jointTorques(0)*teach_k);
                }else if(control_mode == 1 || control_mode==3){
                    CAN_Handlej.Send_DM_Moto(1, 0, 0, 0, 0,  solve.jointTorques(0));
                }
                usleep(200);
        /////////////////////////////torque/////////////////////////////////////
                CAN_Handlej.Can_cmd_all (2, 0, 0, 0, 0,  solve.jointTorques(1)*1);
                usleep(200);
                CAN_Handlej.Can_cmd_all (4, 0, 0, 0, 0,  solve.jointTorques(2)*1);
                usleep(200);
                CAN_Handlej.Send_DM_Moto(5, 0, 0, 0, 0,  solve.jointTorques(3)+0);
                usleep(200);
                CAN_Handlej.Send_DM_Moto(6, 0, 0, 0, 0,  solve.jointTorques(4)+0);
                usleep(200);
                CAN_Handlej.Send_DM_Moto(7, 0, 0, 0, 0,  solve.jointTorques(5)+0);
                usleep(200);

                CAN_Handlej.Send_DM_Moto(8, 0, 0, 0, 0, ros_control_cur_t[6]);
                usleep(200);

            } else {
                    if(play.is_playing)
                    {
                        if(control_mode == 0 || control_mode==2){
                            CAN_Handlej.Can_cmd_all (1, 150, 12, target_pos[0], 0, 0);
                        }else if(control_mode == 1 || control_mode==3){
                            CAN_Handlej.Send_DM_Moto(1, 15, 1, target_pos[0], 0, 0);
                        }
                        usleep(200);
            ////////////////////////////force/////////////////////////////////////
                        CAN_Handlej.Can_cmd_all (2, 150, 12, target_pos[1], 0, 0);
                        usleep(200);
                        CAN_Handlej.Can_cmd_all (4, 150, 12, target_pos[2], 0, 0);
                        usleep(200);
                        CAN_Handlej.Send_DM_Moto(5, 15,  1,  target_pos[3], 0, 0);
                        usleep(200);
                        CAN_Handlej.Send_DM_Moto(6, 15,  1,  target_pos[4], 0, 0);
                        usleep(200);
                        CAN_Handlej.Send_DM_Moto(7, 10,  1,  target_pos[5], 0, 0);
                        usleep(200);
                    }
                    else
                    {
            ////////////////////////////force/////////////////////////////////////
// ros_control_cur_t
//                        if(control_mode == 0 || control_mode==2){
//                             CAN_Handlej.Can_cmd_all (1, 150, 12, target_pos[0], 0, ros_control_cur_t[0]);
//                         }else if(control_mode == 1 || control_mode==3){ 
//                             CAN_Handlej.Send_DM_Moto(1, 39, 0.8, target_pos[0], 0, ros_control_cur_t[0] );
//                         } 
//                         CAN_Handlej.Can_cmd_all (2, 150, 12, target_pos[1],     0, ros_control_cur_t[1] ); //210
//                         usleep(200); 
//                         CAN_Handlej.Can_cmd_all (4, 150, 12, target_pos[2],     0, ros_control_cur_t[2] );
//                         usleep(200); 
//                         CAN_Handlej.Send_DM_Moto(5, 30, 0.8, target_pos[3],     0, ros_control_cur_t[3] ); //20 0.8   25  0.8
//                         usleep(200); 
//                         CAN_Handlej.Send_DM_Moto(6, 25, 0.8, target_pos[4],     0, ros_control_cur_t[4] );
//                         usleep(200); 
//                         CAN_Handlej.Send_DM_Moto(7, 10, 1  , target_pos[5],     0, ros_control_cur_t[5] );
//                         usleep(200); 

                       if(control_mode == 0 || control_mode==2){
                            CAN_Handlej.Can_cmd_all (1, 150, 12, target_pos[0], 0, solve.jointTorques(0));
                        }else if(control_mode == 1 || control_mode==3){
                            CAN_Handlej.Send_DM_Moto(1, 39, 0.8, target_pos[0], 0, solve.jointTorques(0));
                        }
                        CAN_Handlej.Can_cmd_all (2, 150, 12, target_pos[1], 0, solve.jointTorques(1)); //210
                        usleep(200);
                        CAN_Handlej.Can_cmd_all (4, 150, 12, target_pos[2], 0, solve.jointTorques(2));
                        usleep(200);
                        CAN_Handlej.Send_DM_Moto(5, 30, 0.8, target_pos[3], 0, solve.jointTorques(3)); //20 0.8   25  0.8
                        usleep(200);
                        CAN_Handlej.Send_DM_Moto(6, 25, 0.8, target_pos[4], 0, solve.jointTorques(4));
                        usleep(200);
                        CAN_Handlej.Send_DM_Moto(7, 10, 1  , target_pos[5], 0, solve.jointTorques(5));
                        usleep(200);
                        CAN_Handlej.Send_DM_Moto(8, 0, 0, 0, 0, 0);
                        usleep(200);


                    }
                }
    // ROS_INFO("\033[32mangle_joint = 1>%f 2>%f 3>%f 4>%f 5>%f 6>%f 7>%f \033[0m",    current_pos[0], \
    //                                                                                 current_pos[1], \
    //                                                                                 current_pos[2], \
    //                                                                                 current_pos[3], \
    //                                                                                 current_pos[4], \
    //                                                                                 current_pos[5], \
    //                                                                                 current_pos[6]); 

    }else
    {               
                    CAN_Handlej.Can_cmd_all(1, 0, 12, 0, 0, 0);
                    CAN_Handlej.Can_cmd_all(2, 0, 12, 0, 0, 0);
                    CAN_Handlej.Can_cmd_all(4, 0, 12, 0, 0, 0);
                    CAN_Handlej.Send_DM_Moto(5, 0, 1, 0, 0, 0);
                    usleep(200);
                    CAN_Handlej.Send_DM_Moto(6, 0, 1, 0, 0, 0);
                    usleep(200);
                    CAN_Handlej.Send_DM_Moto(7, 0, 1, 0, 0, 0);
                    usleep(200); 
                    ROS_WARN("safe mode!!!!!!!!!");

    }

}

// void arx_arm::gripper_f()
// {
//     CAN_Handlej.Send_DM_Moto(8, 0, 1, 0, 1, 0);
    
// }


void arx_arm::joint_control()
{
        float ramp_ros_pos_k=0.007f;//0.005f
        float ramp_ros_pos_k2=0.01f;//0.005f
        ros_control_pos[0]=ramp( ros_control_pos_t[0], ros_control_pos[0], ramp_ros_pos_k2);    
        ros_control_pos[1]=ramp( ros_control_pos_t[1], ros_control_pos[1], ramp_ros_pos_k);
        ros_control_pos[2]=ramp( ros_control_pos_t[2], ros_control_pos[2], ramp_ros_pos_k);
        ros_control_pos[3]=ramp( ros_control_pos_t[3], ros_control_pos[3], ramp_ros_pos_k2);
        ros_control_pos[4]=ramp( ros_control_pos_t[4], ros_control_pos[4], ramp_ros_pos_k2);
        ros_control_pos[5]=ramp( ros_control_pos_t[5], ros_control_pos[5], ramp_ros_pos_k2);

        if(control_mode == 2){
            CAN_Handlej.Can_cmd_all (1, 150, 12, ros_control_pos[0], 0, 0);
        }else if(control_mode == 3){
            CAN_Handlej.Send_DM_Moto(1, 39, 0.8, ros_control_pos[0], 0, 0); //20 0.8   25  0.8
        }
        usleep(200);
        CAN_Handlej.Can_cmd_all (2, 210, 12, ros_control_pos[1], 0, 0);
        usleep(200);
        CAN_Handlej.Can_cmd_all (4, 150, 12, ros_control_pos[2], 0, 0);
        usleep(200);
        CAN_Handlej.Send_DM_Moto(5, 30, 0.8, ros_control_pos[3], 0, 0); //20 0.8   25  0.8
        usleep(200);
        CAN_Handlej.Send_DM_Moto(6, 25, 0.8, ros_control_pos[4], 0, 0);
        usleep(200);
        CAN_Handlej.Send_DM_Moto(7, 10, 1  , ros_control_pos[5], 0, 0);
        usleep(200); 

    ROS_INFO("\033[32mangle_ ros_joint = 1>%f 2>%f 3>%f 4>%f 5>%f 6>%f 7>%f \033[0m",    ros_control_pos[0], \
                                                                                         ros_control_pos[1], \
                                                                                         ros_control_pos[2], \
                                                                                         ros_control_pos[3], \
                                                                                         ros_control_pos[4], \
                                                                                         ros_control_pos[5], \
                                                                                         ros_control_pos[6]); 

}


void arx_arm::init_step()
{
   temp_init++;
   if(temp_init>2)
   {
            for (int i = 0; i < 6; ++i)
            {
                target_pos[i] = ramp(0, target_pos[i], 0.003); //0.003   0.001
            }
            bool all_positions_within_threshold = true;
            for (int i = 0; i < 6; ++i) {
                if (std::fabs(current_pos[i]) >= 0.1) {
                    all_positions_within_threshold = false;
                    calc_init=0;
                    break;
                }
            }
            if (all_positions_within_threshold) {
                calc_init++;
                if(calc_init>300){
                    is_starting=0;
                    is_arrived=1;
                }

            } else {
                is_arrived =0;
                is_init=2;
                teach2pos_returning = false ;
            }

            if(init_kp<150)
            init_kp+= 1.0f; //1.0f
            if(init_kd< 12) 
            init_kd+=0.2f;  //0.2
            if(init_kp_4<80) 
            init_kp_4+=0.3f; 
            if(init_kp_4>=80 < 150)
            init_kp_4+= 0.5f; 
            if(init_kd_4<5)
            init_kd_4+=0.1f;
            if(init_kd_6<0.45)
            init_kd_6+=0.1f;


            if(control_mode == 0 || control_mode == 2){
                CAN_Handlej.Can_cmd_all (1, init_kp, init_kd, target_pos[0], 0,  0);  //150  12
            }else if(control_mode == 1 || control_mode == 3){
                CAN_Handlej.Send_DM_Moto(1, 20, 1, target_pos[0], 0, 0);
            }  
            usleep(200);          
            CAN_Handlej.Can_cmd_all (2, init_kp, init_kd, target_pos[1], 0,  0);
            usleep(200);
            CAN_Handlej.Can_cmd_all (4, init_kp, init_kd, target_pos[2], 0,  0);  //init_kp_4
            usleep(200);
            CAN_Handlej.Send_DM_Moto(5, 30, 0.8, target_pos[3], 0, 0);
            usleep(200);
            CAN_Handlej.Send_DM_Moto(6, 25, 0.8, target_pos[4], 0, 0);
            usleep(200);
            CAN_Handlej.Send_DM_Moto(7, 10, 1, target_pos[5], 0, 0);
            usleep(200); 

   }
    else
    {
        for (int i = 0; i < 6; ++i)
        {
            target_pos[i] = current_pos[i];
        }
    }
    ROS_WARN(">>>is_init>>>");

    // ROS_INFO("\033[32mangle_joint = 1>%f 2>%f 3>%f 4>%f 5>%f 6>%f 7>%f starting>%d init_kp>%f init_kd>%f \033[0m",    current_pos[0], \
    //                                                                                 current_pos[1], \
    //                                                                                 current_pos[2], \
    //                                                                                 current_pos[3], \
    //                                                                                 current_pos[4], \
    //                                                                                 current_pos[5], \
    //                                                                                 current_pos[6],\
    //                                                                                 is_starting,\ 
    //                                                                                 init_kp,\     
    //                                                                                 init_kd\                                                                            
                                                                                    
    //                                                                                 ); 


      
}


int arx_arm::rosGetch()
{ 
    static struct termios oldTermios, newTermios;
    tcgetattr( STDIN_FILENO, &oldTermios);          
    newTermios = oldTermios; 
    newTermios.c_lflag &= ~(ICANON);                      
    newTermios.c_cc[VMIN] = 0; 
    newTermios.c_cc[VTIME] = 0;
    tcsetattr( STDIN_FILENO, TCSANOW, &newTermios);  

    int keyValue = getchar(); 

    tcsetattr( STDIN_FILENO, TCSANOW, &oldTermios);  
    return keyValue;
}


void arx_arm::getKey(char key_t) {
   int wait_key=100;

    if(key_t == 'w')
    arx5_cmd.key_x = arx5_cmd.key_x_t=1;
    else if(key_t == 's')
    arx5_cmd.key_x = arx5_cmd.key_x_t=-1;
    else arx5_cmd.key_x_t++;
    if(arx5_cmd.key_x_t>wait_key)
    arx5_cmd.key_x = 0;

    if(key_t == 'a')
    arx5_cmd.key_y =arx5_cmd.key_y_t= 1;
    else if(key_t == 'd')
    arx5_cmd.key_y =arx5_cmd.key_y_t= -1;
    else if(key_t == 'R')
    arx5_cmd.key_y =arx5_cmd.key_y_t= -1;
    else if(key_t == 'L')
    arx5_cmd.key_y =arx5_cmd.key_y_t= 1;
    else arx5_cmd.key_y_t++;
    if(arx5_cmd.key_y_t>wait_key)
    arx5_cmd.key_y = 0;   

    if(key_t == 'U')
    arx5_cmd.key_z =arx5_cmd.key_z_t= 1;
    else if(key_t == 'D')
    arx5_cmd.key_z =arx5_cmd.key_z_t= -1;
    else arx5_cmd.key_z_t++;
    if(arx5_cmd.key_z_t>wait_key)
    arx5_cmd.key_z = 0;

    if(key_t == 'q')
    arx5_cmd.key_base_yaw =arx5_cmd.key_base_yaw_t= 1;
    else if(key_t == 'e')
    arx5_cmd.key_base_yaw =arx5_cmd.key_base_yaw_t= -1;
    else arx5_cmd.key_base_yaw_t++;
    if(arx5_cmd.key_base_yaw_t>wait_key)
    arx5_cmd.key_base_yaw = 0;

    if(key_t == 'r')
    arx5_cmd.key_reset =arx5_cmd.key_reset_t=1;
    else arx5_cmd.key_reset_t++;
    if(arx5_cmd.key_reset_t>wait_key)
    arx5_cmd.key_reset =0;

    if(key_t == 'i')
    arx5_cmd.key_i =arx5_cmd.key_i_t=1;
    else arx5_cmd.key_i_t++;
    if(arx5_cmd.key_i_t>wait_key)
    arx5_cmd.key_i =0;

    if(key_t == 'p')
    arx5_cmd.key_p =arx5_cmd.key_p_t=1;
    else arx5_cmd.key_p_t++;
    if(arx5_cmd.key_p_t>wait_key)
    arx5_cmd.key_p =0;

    if(key_t == 'o')
    arx5_cmd.key_o =arx5_cmd.key_o_t=1;
    else arx5_cmd.key_o_t++;
    if(arx5_cmd.key_o_t>wait_key)
    arx5_cmd.key_o =0;

    if(key_t == 'c')
    arx5_cmd.key_c =arx5_cmd.key_c_t=1;
    else arx5_cmd.key_c_t++;
    if(arx5_cmd.key_c_t>wait_key)
    arx5_cmd.key_c =0;

    if(key_t == 't')
    arx5_cmd.key_t =arx5_cmd.key_t_t=1;
    else arx5_cmd.key_t_t++;
    if(arx5_cmd.key_t_t>wait_key)
    arx5_cmd.key_t =0;

    if(key_t == 'g')
    arx5_cmd.key_g =arx5_cmd.key_g_t=1;
    else arx5_cmd.key_g_t++;
    if(arx5_cmd.key_g_t>wait_key)
    arx5_cmd.key_g =0; 

    if(key_t == 'm')
    arx5_cmd.key_m =arx5_cmd.key_m_t=1;
    else arx5_cmd.key_m_t++;
    if(arx5_cmd.key_m_t>wait_key)
    arx5_cmd.key_m =0;
// r p y
    if(key_t == 'n')
    arx5_cmd.key_roll =arx5_cmd.key_roll_t= 1;
    else if(key_t == 'm')
    arx5_cmd.key_roll =arx5_cmd.key_roll_t= -1;
    else arx5_cmd.key_roll_t++;
    if(arx5_cmd.key_roll_t>wait_key)
    arx5_cmd.key_roll = 0;  

    if(key_t == 'l')
    arx5_cmd.key_pitch =arx5_cmd.key_pitch_t= 1;
    else if(key_t == '.')
    arx5_cmd.key_pitch =arx5_cmd.key_pitch_t= -1;
    else arx5_cmd.key_pitch_t++;
    if(arx5_cmd.key_pitch_t>wait_key)
    arx5_cmd.key_pitch = 0;  

    if(key_t == ',')
    arx5_cmd.key_yaw =arx5_cmd.key_yaw_t= 1;
    else if(key_t == '/')
    arx5_cmd.key_yaw =arx5_cmd.key_yaw_t= -1;
    else arx5_cmd.key_yaw_t++;
    if(arx5_cmd.key_yaw_t>wait_key)
    arx5_cmd.key_yaw = 0;  

                
    // ROS_INFO("thekey_is >>%d",key_t);






//chassis
    if(key_t == 'u')
    arx5_cmd.key_u =arx5_cmd.key_u_t=1;
    else arx5_cmd.key_u_t++;
    if(arx5_cmd.key_u_t>wait_key)
    arx5_cmd.key_u =0;

    if(key_t == 'j')
    arx5_cmd.key_j =arx5_cmd.key_j_t=1;
    else arx5_cmd.key_j_t++;
    if(arx5_cmd.key_j_t>wait_key)
    arx5_cmd.key_j =0;

    if(key_t == 'h')
    arx5_cmd.key_h =arx5_cmd.key_h_t=1;
    else arx5_cmd.key_h_t++;
    if(arx5_cmd.key_h_t>wait_key)
    arx5_cmd.key_h =0;

    if(key_t == 'k')
    arx5_cmd.key_k =arx5_cmd.key_k_t=1;
    else arx5_cmd.key_k_t++;
    if(arx5_cmd.key_k_t>wait_key)
    arx5_cmd.key_k =0;

    if(key_t == 'v')
    arx5_cmd.key_v =arx5_cmd.key_v_t=1;
    else arx5_cmd.key_v_t++;
    if(arx5_cmd.key_v_t>wait_key)
    arx5_cmd.key_v =0;

    if(key_t == 'b')
    arx5_cmd.key_b =arx5_cmd.key_b_t=1;
    else arx5_cmd.key_b_t++;
    if(arx5_cmd.key_b_t>wait_key)
    arx5_cmd.key_b =0;
    





    return ;
}


bool arx_arm::modifyLinkMass(const std::string& inputFilePath, const std::string& outputFilePath, double newMassValue) {
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError eResult = doc.LoadFile(inputFilePath.c_str());  // 加载URDF文件

    if (eResult != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to load the file!" << std::endl;
        return false;
    }

    // 查找名为 "link6" 的link元素
    tinyxml2::XMLElement* linkElement = doc.FirstChildElement("robot")->FirstChildElement("link");

    while (linkElement != nullptr) {
        const char* name = linkElement->Attribute("name");
        if (name != nullptr && std::string(name) == "link6") {
            tinyxml2::XMLElement* massElement = linkElement->FirstChildElement("inertial")->FirstChildElement("mass");
            if (massElement != nullptr) {
                massElement->SetAttribute("value", std::to_string(newMassValue).c_str());  // 设置新的质量值
            }
            break;
        }
        linkElement = linkElement->NextSiblingElement("link");
    }

    // 保存修改后的URDF文件
    doc.SaveFile(outputFilePath.c_str());
    return true;
}


void arx_arm::gripper_control()
{
            //速度闭环版本
            // if(Teleop_Use()->axes_[2]<0  || arx5_cmd.key_o) //开启
            // gripper_spd=45;
            // else if (Teleop_Use()->buttons_[4] == 1 || arx5_cmd.key_c) //夹取
            // gripper_spd=-45;
            // gripper_cout=12*(gripper_spd-rv_motor_msg[8].gripper_spd);

            // if(gripper_cout>gripper_max_cur)
            //     gripper_cout=gripper_max_cur;
            // else if(gripper_cout<-gripper_max_cur)
            //     gripper_cout=-gripper_max_cur;
            // CAN_Handlej.CAN_GRIPPER(gripper_cout, 0, 0, 0);
            CAN_Handlej.CAN_GRIPPER(0, 0, 0, 0);
            ////纯电流版本
            // if(Teleop_Use()->axes_[2]<0  || arx5_cmd.key_o) //开启
            // CAN_Handlej.CAN_GRIPPER(36, 0, 0, 0);  
            // else if (Teleop_Use()->buttons_[4] == 1 || arx5_cmd.key_c) //夹取
            // CAN_Handlej.CAN_GRIPPER(-36, 0, 0, 0);  //-45   
            // CAN_Handlej.CAN_GRIPPER(0, 0, 0, 0);  //-45   


}


void arx_arm::arm_torque_mode()
{
        if( (Teleop_Use()->buttons_[5] == 1 && !button5_pressed)  || ( arx5_cmd.key_i ==1 &&  !button5_pressed) )
        {
            button5_pressed = true;
            if (!is_torque_control) 
            {
                init_kp=10,init_kp_4=20,init_kd=init_kd_4=init_kd_6=init_kp_4=0; 
                is_teach_mode = true;
                is_torque_control = true;
                for (int i = 0; i < 6; i++)
                {
                    prev_target_pos[i] = target_pos[i];
                }
            }else
            {   
                is_teach_mode = false;
                is_torque_control = false;
                teach2pos_returning = true;
                for (int i = 0; i < 6; i++)
                {
                    target_pos[i] = current_pos[i];
                }
                
            }
        }
           
 
        if (button5_pressed && (!Teleop_Use()->buttons_[5] && !arx5_cmd.key_i ))
            button5_pressed = false; 

}

void arx_arm::arm_replay_mode()
{
 // Replay 
    if ((Teleop_Use()->buttons_[0] == 1 && !button0_pressed) ||  (  arx5_cmd.key_p ==1  && !button0_pressed) )
    {
            if(!play.is_playing)
            {
            if(play_flag==0 ){
                    is_starting=1;
                    init_kp=0,init_kp_4=0,init_kd=init_kd_4=init_kd_6=init_kp_4=0; 
                    play_flag=2;

                arx5_cmd.reset = true;
                arx5_cmd.waist_pitch  = arx5_cmd.waist_pitch_t  =arx5_cmd.control_pit   = joy_pitch_t  =joy_pitch      =0      ;
                arx5_cmd.x            = arx5_cmd.x_t            =arx5_cmd.control_x     = joy_x_t      =joy_x          =0      ;
                arx5_cmd.y            = arx5_cmd.y_t            =arx5_cmd.control_y     = joy_y_t      =joy_y          =0      ;
                arx5_cmd.z            = arx5_cmd.z_t            =arx5_cmd.control_z     = joy_z_t      =joy_z          =0      ;      
                arx5_cmd.base_yaw     = arx5_cmd.base_yaw_t     =      0 ;
                arx5_cmd.gripper_roll = arx5_cmd.gripper_roll_t =arx5_cmd.control_roll  = joy_roll_t   =joy_roll       =0      ;
                arx5_cmd.waist_yaw    = arx5_cmd.waist_yaw_t    =arx5_cmd.control_yaw   = joy_yaw_t    =joy_yaw        =0      ;
                arx5_cmd.mode = FORWARD;                

                }else{//执行动作
                    play_file_list=play.getFilesList(ros::package::getPath("arm_control") + "/saved_record");
                    play.play_start_all(target_pos,current_pos,play_file_list);
                    play.repeat_stop_flag = false;
        
                }  
            }
            else
            {
                play.repeat_stop_flag = true;
            }
        button0_pressed = true;
        ROS_ERROR("button0_pressed\n");
    }
    if (button0_pressed && (!Teleop_Use()->buttons_[0] && !arx5_cmd.key_p))
    {
        // play_flag=0;
        button0_pressed = false;
        ROS_ERROR("button0_pressed false\n");
    }

}

void arx_arm::arm_reset_mode(){
///////////////////////////////////////////////////////////
        if((Teleop_Use()->buttons_[1] == 1)|| (arx5_cmd.key_reset==1)) // reset
        {      
            arx5_cmd.base_yaw_t=ramp(0.0, arx5_cmd.base_yaw_t, 0.01);    
            arx5_cmd.gripper_t=0;      
            arx5_cmd.control_roll=ramp(0.0, arx5_cmd.control_roll, 0.1); 
            arx5_cmd.control_pit=ramp(0.0, arx5_cmd.control_pit, 0.1);
            arx5_cmd.control_yaw=ramp(0.0, arx5_cmd.control_yaw, 0.1);   

            arx5_cmd.control_x=ramp(0.0, arx5_cmd.control_x, 0.0006); 

            if(arx5_cmd.x <0.01){
            arx5_cmd.control_y=ramp(0.0, arx5_cmd.control_y, 0.0006);
            arx5_cmd.control_z=ramp(0.0, arx5_cmd.control_z, 0.0006);
            }
            joy_yaw=joy_pitch=joy_roll=0;
        }
    
            arx5_cmd.reset = false;

}

void arx_arm::arm_get_pos(){

            arx5_cmd.base_yaw_t += (Teleop_Use()->axes_[0]/100.0f + arx5_cmd.key_base_yaw/100.0f);
            // ROS_INFO("arx5_cmd.base_yaw_t>%f,Teleop_Use()->axes_[0]>%f,arx5_cmd.key_base_yaw>%f",arx5_cmd.base_yaw_t,Teleop_Use()->axes_[0],arx5_cmd.key_base_yaw);
            // 手柄通道+键盘通道+ROS通道  
            if(abs(arx5_ros_cmd.x)<0.1)
                ros_move_k_x=500;
            else ros_move_k_x=100;

            if(abs(arx5_ros_cmd.y)<0.1)
                ros_move_k_y=500;
            else ros_move_k_y=100;

            if(abs(arx5_ros_cmd.z)<0.1)
                ros_move_k_z=500;
            else ros_move_k_z=100;

            arx5_cmd.control_x += (joystick_projection(Teleop_Use()->axes_[1])/1000.0f+arx5_cmd.key_x/2000.0f  +arx5_ros_cmd.x/ros_move_k_x);
            arx5_cmd.control_y += (joystick_projection(Teleop_Use()->axes_[3])/1000.0f+arx5_cmd.key_y/2000.0f  +arx5_ros_cmd.y/ros_move_k_y);
            arx5_cmd.control_z += (joystick_projection(Teleop_Use()->axes_[4])/1000.0f+arx5_cmd.key_z/2000.0f  +arx5_ros_cmd.z/ros_move_k_z);

            if (Teleop_Use()->buttons_[2] == 1){ //手柄控制 - 键位组合
            arx5_cmd.control_pit -= (Teleop_Use()->axes_[7]/100.0f+arx5_cmd.key_pitch/1000.0f);
            arx5_cmd.control_yaw   += (Teleop_Use()->axes_[6]/100.0f+arx5_cmd.key_yaw/1000.0f);
            }else
            { //arx5_ros_cmd
            arx5_cmd.control_pit += (arx5_ros_cmd.pitch/1000.0f -arx5_cmd.key_pitch/100.0f);
            arx5_cmd.control_yaw   += (arx5_ros_cmd.yaw/1000.0f   +arx5_cmd.key_yaw  /100.0f);
            }
            arx5_cmd.control_roll  += (-Teleop_Use()->axes_[6]/100.0f-arx5_cmd.key_roll/100.0f + arx5_ros_cmd.roll/1000.0f);
            
            joy_x_t = arx5_cmd.control_x      + magic_pos[0];
            joy_y_t = arx5_cmd.control_y      + magic_pos[1];
            joy_z_t = arx5_cmd.control_z      + magic_pos[2];
            joy_pitch_t=arx5_cmd.control_pit  + magic_angle[0];
            joy_yaw_t  =arx5_cmd.control_yaw  + magic_angle[1];
            joy_roll_t =arx5_cmd.control_roll + magic_angle[2];
            
            //限位
            limit_pos();

            arx5_cmd.reset = true;
            float reset_temp_k=0.001;

                arx5_cmd.x            = ramp(joy_x_t, arx5_cmd.x, reset_temp_k);  
                arx5_cmd.y            = ramp(joy_y_t, arx5_cmd.y, reset_temp_k);
                arx5_cmd.z            = ramp(joy_z_t, arx5_cmd.z, reset_temp_k);
                arx5_cmd.base_yaw     = ramp(arx5_cmd.base_yaw_t, arx5_cmd.base_yaw, 0.009);
                arx5_cmd.gripper_roll = ramp(joy_roll_t, arx5_cmd.gripper_roll, 0.01);
                arx5_cmd.waist_pitch  = ramp(joy_pitch_t, arx5_cmd.waist_pitch, 0.01);
                arx5_cmd.waist_yaw    = ramp(joy_yaw_t, arx5_cmd.waist_yaw, 0.01);
                arx5_cmd.mode = FORWARD;
   

}

void arx_arm::arm_teach_mode(){

// Record settings
            if (arx5_cmd.key_t == 1 && !button_teach)
            {
                if(!is_recording){
                    is_recording  = true;
                    is_teach_mode = true;
                }
                else
                {
                    is_recording  = false;
                    is_teach_mode = false;
                    out_teach_path = ros::package::getPath("arm_control") + "/teach_record/out.txt";
                    play.end_record(out_teach_path);
                    // current_normal = false;
                }
                button_teach = true;
            }
            if (button_teach && !arx5_cmd.key_t)
                button_teach = false;

            // Replay 
            if ((arx5_cmd.key_g == 1) && (!is_recording) && (!button_teach))
            {
                std::vector<std::string> arx;
                    play.play_start(ros::package::getPath("arm_control") + "/teach_record/out.txt",target_pos,current_pos,arx);
                button_replay = true;
                // ROS_ERROR("debug");
            }
            if (button_replay && !arx5_cmd.key_g)
                button_replay = false;
 
}

void arx_arm::limit_pos()
{
        joy_x_t = limit<float>(joy_x_t, lower_bound_waist[0], upper_bound_waist[0]);
        joy_y_t = limit<float>(joy_y_t, lower_bound_waist[1], upper_bound_waist[1]);
        joy_z_t = limit<float>(joy_z_t, lower_bound_waist[2], upper_bound_waist[2]);
        joy_pitch_t = limit<float>(joy_pitch_t, lower_bound_pitch, upper_bound_pitch);
        joy_yaw_t   = limit<float>(joy_yaw_t, lower_bound_yaw, upper_bound_yaw);
        joy_roll_t  = limit<float>(joy_roll_t, lower_bound_sim[ROLL], upper_bound_sim[ROLL]);

        arx5_cmd.control_x = limit<float>(arx5_cmd.control_x, lower_bound_waist[0], upper_bound_waist[0]);
        arx5_cmd.control_y = limit<float>(arx5_cmd.control_y, lower_bound_waist[1], upper_bound_waist[1]);
        arx5_cmd.control_z = limit<float>(arx5_cmd.control_z, lower_bound_waist[2], upper_bound_waist[2]);
        arx5_cmd.control_pit   = limit<float>(arx5_cmd.control_pit, lower_bound_pitch, upper_bound_pitch);
        arx5_cmd.control_yaw   = limit<float>(arx5_cmd.control_yaw, lower_bound_yaw, upper_bound_yaw);
        arx5_cmd.control_roll  = limit<float>(arx5_cmd.control_roll, lower_bound_sim[ROLL], upper_bound_sim[ROLL]);

        magic_pos[0] = limit<float>(magic_pos[0], lower_bound_waist[0], upper_bound_waist[0]);
        magic_pos[1] = limit<float>(magic_pos[1], lower_bound_waist[1], upper_bound_waist[1]);
        magic_pos[2] = limit<float>(magic_pos[2], lower_bound_waist[2], upper_bound_waist[2]);
        magic_angle[0] = limit<float>(magic_angle[0], lower_bound_pitch, upper_bound_pitch);
        magic_angle[1] = limit<float>(magic_angle[1], lower_bound_yaw, upper_bound_yaw);
        magic_angle[2] = limit<float>(magic_angle[2], lower_bound_sim[ROLL], upper_bound_sim[ROLL]);        

}

void arx_arm::cur_change()
{
    for (int i = 0; i < 6; i++)
    {
        if(control_mode == 0 || control_mode ==2){  //arx5
            if(i>2){
                solve.jointTorques(i) = solve.jointTorques(i) * 2.36; //2.45  2.3 
            }else{solve.jointTorques(i) = solve.jointTorques(i) * 0.72f; }            
        }else if(control_mode == 1 || control_mode ==3)  //5a
        {
            if(i>2){
                solve.jointTorques(i) = solve.jointTorques(i) * 2.36; //2.45  2.3 
            }else{solve.jointTorques(i) = solve.jointTorques(i) * 0.72f; }   
        }


        if (solve.jointTorques(i) >= max_torque)
            solve.jointTorques(i) = max_torque;
        else if (solve.jointTorques(i) <= -max_torque)
            solve.jointTorques(i) = -max_torque;
        
        solve.jointTorques(i) = solve.jointTorques(i) * (1.0 - filter_torque) + solve.jointPrevTorques(i) * filter_torque;
        solve.jointPrevTorques(i) = solve.jointTorques(i);
    }


    // for (int i = 0; i < 6; i++)
    // {
    //     if(control_mode == 0 || control_mode ==2){  //arx5
    //         if(i>2){
    //             ros_control_cur_t(i) = ros_control_cur_t * 2.36; //2.45  2.3 
    //         }else{solve.jointTorques(i) = solve.jointTorques(i) * 0.72f; }            
    //     }else if(control_mode == 1 || control_mode ==3)  //5a
    //     {
    //         if(i>2){
    //             solve.jointTorques(i) = solve.jointTorques(i) * 2.36; //2.45  2.3 
    //         }else{solve.jointTorques(i) = solve.jointTorques(i) * 0.72f; }   
    //     }


    //     if (solve.jointTorques(i) >= max_torque)
    //         solve.jointTorques(i) = max_torque;
    //     else if (solve.jointTorques(i) <= -max_torque)
    //         solve.jointTorques(i) = -max_torque;
        
    //     solve.jointTorques(i) = solve.jointTorques(i) * (1.0 - filter_torque) + solve.jointPrevTorques(i) * filter_torque;
    //     solve.jointPrevTorques(i) = solve.jointTorques(i);
    // }

}

