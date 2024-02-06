#ifndef _CAN_H_
#define _CAN_H_

#include <stdint.h>
#include <string.h>
#include <iostream>
// #include <ros/ros.h>
// #include "sensor_msgs/Imu.h"
// #include "tf/tf.h"
#include "../libcan/SocketCAN.h"
#include "motor.h"

// #include "arm_control/Imu.h"
// #include "arm_control/Motor.h"

#define LEFT_WHEEL 1
#define RIGHT_WHEEL 0
#define MOTOR_NUM 8 // number of motors

// 每个电机的速度
/**
 * @brief motor measurement backward
 *
 * @param angle: [-18000, 18000] in int VS [-180, 180] in degrees
 * @param speed_rpm: [-32768, 32767] in int VS [-3276.8, 3276.7] in rpm
 * @param current_actual: [-32768, 32767] in int VS [-327.68, 327.67] in A @TODO：A????????
 */
// typedef struct
// {
//     int16_t speed_rpm;
//     int16_t real_current;
//     int16_t given_current;
//     int16_t angle;      
//     int16_t last_angle; 
//     int16_t offset_angle;
//     int32_t total_angle;
//     int32_t round_cnt;
//     uint32_t msg_cnt;
// } Motor_measure_t;

// //所有角度信息 角度和角加速度
typedef struct
{
    float imu[3];  //单位 rad    顺序yaw pitch roll
    float last_imu[3]; //上一时刻的imu角度数据
    float gyro[3]; //单位 rad/s  顺序pitch  roll  yaw
} IMU_Float_t;

typedef struct{
    int8_t   temperature;
    int16_t	 speed_rpm;
    int16_t  real_current;
    uint16_t position;
    int8_t   round_cnt;
    float    total_angle;
    float    total_angle_last;
	
}m_rmd_t;






// 负责can收发
class can
{
public:
    can();
    ~can();

    // sensor_msgs::Imu ImuMsg;
    // Motor_measure_t Motor[MOTOR_NUM] = {0};

    void can3_ReceiveFrame(can_frame_t *frame);

    void CAN_cmd_readMotorID(void);
    void CAN_cmd_getMotorParam(uint16_t motor_id, uint8_t param_cmd);
    void CAN_cmd_init(uint16_t motor_id,uint8_t cmd);
    void Can_cmd_position(uint16_t motor_id,float pos,uint16_t spd,uint16_t cur,uint8_t ack_status);

    void Can_cmd_all(uint16_t motor_id, float kp, float kd, float pos, float spd, float tor);
    // motor id 0x01 ~ 0x04
    void CAN_cmd_fpc1(float kp[4], float kd[4], float pos[4], float spd[4], float tor[4]);
    // motor id 0x05 ~ 0x08
    void CAN_cmd_fpc2(float kp[4], float kd[4], float pos[4], float spd[4], float tor[4]);
    void MotorSetting(uint16_t motor_id,uint8_t cmd);
    // // motor id 0x01 ~ 0x04
    // void CAN_cmd_chassis1(int16_t motor[4]);
    // // motor id 0x05 ~ 0x08
    // void CAN_cmd_chassis2(int16_t motor[4]);
    
    void GetImuAngle(uint8_t *data);
    void GetImuGyro(uint8_t *data);
    // // int freq = 0;
    void CAN_GRIPPER(int16_t motor1, int16_t motor2, int16_t motor3, int16_t motor4);
    //dubuf DM
    void Send_DM_Moto(uint16_t motor_id, float kp, float kd, float pos, float spd, float tor);
    void Enable_Moto(uint16_t ID);
    void Set_Zero(uint16_t ID);
    void clear(uint16_t ID);

private:
   
    SocketCAN can3_adapter;

};

#endif