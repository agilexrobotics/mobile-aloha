#include "Hardware/teleop.h"

Teleop *Teleop_Hardware = NULL;

Teleop *Teleop_Use()
{
    if(Teleop_Hardware == NULL)
    {
        Teleop_Hardware = new Teleop();
    }
    return Teleop_Hardware;
}

void Release_Teleop_Hardware()
{
    if(Teleop_Hardware != NULL)
    {
        delete Teleop_Hardware;
        Teleop_Hardware = NULL;
    }
    return;
}

void Teleop::teleop_init(ros::NodeHandle& nh)
{
    // nh_ = nh;
    joy_sub_ = nh.subscribe<sensor_msgs::Joy>("joy", 10, &Teleop::joyCallback, this);
    //sony joystick
    axes_[2] = 1.0;
    axes_[5] = 1.0;
}

void Teleop::joyCallback(const sensor_msgs::Joy::ConstPtr &joy)
{
    rv_time_ = clock();
    for (int i = 0; i < 8; i++)
    {
        axes_[i] = joy->axes[i];
    }
    for (int i = 0; i < 13; i++) // sony 12 else 10 buttons
    {
        buttons_[i] = joy->buttons[i];
    }
}
