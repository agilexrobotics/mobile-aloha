#ifndef _TELEOP_H_
#define _TELEOP_H_

#include <ctime>
#include <ros/ros.h>
#include <sensor_msgs/Joy.h>

class Teleop
{
public:
    Teleop() = default;
    ~Teleop() = default;
    void teleop_init(ros::NodeHandle& nh);
    double axes_[8] = {0.0};
    int buttons_[11] = {0};
    clock_t rv_time_ = clock();
private:
    void joyCallback(const sensor_msgs::Joy::ConstPtr &joy);
    // ros::NodeHandle& nh_;
    ros::Subscriber joy_sub_;
};

Teleop *Teleop_Use();
void Release_Teleop_Hardware();

#endif
