#ifndef _play_H_
#define _play_H_
  
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <cmath> 

#include <sstream>
#include <string>

struct cmd
{

    bool is_recording = false;
    bool is_playing = false;
    bool is_returning = false; 

};

class arx5_play
{

public:
    arx5_play(void);
    ~arx5_play()=default;


    int current_file_num ;
    std::vector<std::string> file_list;

    bool repeat_record = true;
    bool have_replayed = false;

    std::ifstream record_stream;
    bool is_recording = false;
    bool is_playing= false;
    bool is_play_initing=true;
    bool is_returning = false; // flag for indicating whether the robot is returning to start position of the replay
    bool repeat_stop_flag = false;


    // void end_record();
    void end_record(std::string out_path);

    // void update_record(float *current_pos);
    void update_record(float *current_pos,float* solve_pos);
    std::string saved_record;


    float replay_start_pos[7]; // base_yaw x y z r p y
    float start_target_pos[7]; //6 debug
    float replay_joint_pos_temp[7];
    // bool read_write_multiple_files = true;
    bool is_torque_control = false;
    bool is_teach_mode = false;



    cmd play_cmd;
    std::vector<std::string> getFilesList(std::string dirpath);
    void play_start_all(float *input_pos,float *temp_pos,std::vector<std::string> file_list_t);
    void play_start(std::string path,float *input_pos,float *temp_pos,std::vector<std::string> file_list_t);
    void replay_step(float *input_pos,float *temp_pos,std::vector<std::string> file_list_t);
    void init_step(float *get_pos,float *set_pos);


    float ramp(float goal, float current, float ramp_k);


    int play_calc_init=0;
    int init_calc_init=0;
    void read_init(std::vector<std::string> init_read);
    void read_end(std::vector<std::string> init_read);
    float read_init_pos[7]={};
    float read_end_pos[7]={};

};


#endif