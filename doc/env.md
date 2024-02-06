# 1 ACT环境配置

+ 本机基本环境Ubuntu-20.04、CUDA-11.3

# 1 env
~~~python
# 1 拉取源码mobile-aloha
git clone https://github.com/agilexrobotics/mobile-aloha.git

# 2 创建虚拟环境
conda create -n aloha python=3.8

# 3 激活虚拟环境
conda activate aloha

# 4 安装适合cuda的torch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


# 5 安装act依赖
cd mobile-aloha/act

## 5.1 安装requirements依赖
pip install -r requirements.txt

## 5.2 安装detr
cd detr
pip install -v -e .
~~~

# 2 run-act


1. 生成仿真数据record_sim_episodes.py

~~~python
cd act

# 1 格式
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir <data save dir> --num_episodes 50
# --dataset_dir 保存路径

# 2.1 运行 --task_name sim_transfer_cube_scripted
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data/sim_transfer_cube_scripted --num_episodes 10
# 2.2 运行 --task_name sim_insertion_scripted
python3 record_sim_episodes.py --task_name sim_insertion_scripted --dataset_dir data/sim_tran
sfer_cube_scripted --num_episodes 10

# 2.1 实时渲染--onscreen_render参数
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data/sim_transfer_cube_scripted --num_episodes 10  --onscreen_render
~~~

2. visualize episodes

~~~python
# 1 格式
python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

# 2 运行
python3 visualize_episodes.py --dataset_dir data/sim_transfer_cube_scripted --episode_idx 9
# --episode_idx 场景索引号
# 终端打印
# Saved video to: data/sim_transfer_cube_scripted/episode_0_video.mp4
# Saved qpos plot to: data/sim_transfer_cube_scripted/episode_0_qpos.png

~~~

3. imitate episodes-train & eval

~~~python
# --onscreen_render 实时渲染
# --eval            评估
# --num_epochs      训练周期
# --ckpt_dir        权重保存路径 --eval时trainings/policy_best.ckpt文件必须存在

#  1. 训练
python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir trainings --policy_class ACT --kl_weight 1 --chunk_size 10 --hidden_dim 512 --dim_feedforward 3200  --lr 1e-5 --seed 0 --batch_size 8 --num_epochs 100 --onscreen_render

# 2 评估eval
python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir trainings --policy_class ACT --kl_weight 1 --chunk_size 10 --hidden_dim 512 --dim_feedforward 3200  --lr 1e-5 --seed 0 --batch_size 8 --num_epochs 100 --onscreen_render --eval
~~~


~~~python
 File "imitate_episodes.py", line 103, in main
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)
...
FileNotFoundError: [Errno 2] Unable to open file (unable to open file: name = '<put your data dir here>
~~~


~~~
AttributeError: module 'em' has no attribute 'RAW_OPT'
pip install empy==3.3.4
~~~