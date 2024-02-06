# 训练代码imitate_episodes.py

+ **ACT** Action Chunking with Transformers

+ 一个生成式模型，以CVAE（conditional variational autoencoder）的形式来训练模型，根据输入观测值，生成预测动作。


# 1 数据处理

+ 在训练过程中，从臂的关节角和图像一起组成的输入给模型的上一次观测值，而主臂的关节角则作为了动作标签下一次的。
+ 所以采集数据包含 主臂的动作，从臂qpos、images


+ class ACTPolicy(nn.Module)类中 
~~~python
# 1 class ACTPolicy(nn.Module)类中 
# 从下面可以看qpos, image为输入， actions为标签
a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
loss_dict = dict()
all_l1 = F.l1_loss(actions, a_hat, reduction='none')

# 2 class CNNMLPPolicy(nn.Module)类中
actions = actions[:, 0]   # 动作                         # 主臂的action为标签
a_hat = self.model(qpos, image, env_state, actions)     #  从臂的qpos, image为输入x
# 均方误差
mse = F.mse_loss(actions, a_hat)

loss_dict = dict()
loss_dict['mse'] = mse
loss_dict['loss'] = loss_dict['mse']

# 3 self.model() 模型函数  喂入参数actions主要是判断是否含有标签train or val
"""
qpos: batch, qpos_dim
image: batch, num_cam, channel, height, width
env_state: None
actions: batch, seq, action_dim
"""
~~~


## 1.1 image图像数据

~~~python
def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image
~~~

## 1.2 qpos

~~~python
def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    ...
    # construct dataset and dataloader 归一化处理  结构化处理数据
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)

# EpisodicDataset函数
qpos = root['/observations/qpos'][start_ts]
qvel = root['/observations/qvel'][start_ts]
image_dict = dict()
action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
~~~



支持sim和real训练

1. 加载数据集
2. 数据集预处理  (归一化, 随机打乱, 批处理, 模型的需要的类型)
3. 训练，验证    (每100个周期，最后一次，验证集损失最低时的权重)

~~~PYTHON
policy_config = {'lr': args['lr'],
                'num_queries': args['chunk_size'],
                'kl_weight': args['kl_weight'],
                'hidden_dim': args['hidden_dim'],      # 隐藏维度
                'dim_feedforward': args['dim_feedforward'],
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'enc_layers': enc_layers,                 # 编码器层数
                'dec_layers': dec_layers,                 # 解码器层数
                'nheads': nheads,                         # 多头注意力机制的头数
                'camera_names': camera_names,             # 相机名字
                }
config = {
    'num_epochs': num_epochs,                             #周期
    'ckpt_dir': ckpt_dir,
    'episode_len': episode_len,                           # 回合
    'state_dim': state_dim,                               # 状态维度
    'lr': args['lr'],                                     # 学习率
    'policy_class': policy_class,   #   上面的参数          # 模型类型 CVT CNNMLP
    'onscreen_render': onscreen_render,                   # 屏幕渲染
    'policy_config': policy_config,                       # 模型的配置
    'task_name': task_name,                             
    'seed': args['seed'],
    'temporal_agg': args['temporal_agg'],
    'camera_names': camera_names,
    'real_robot': not is_sim
}
~~~






# ACT: Action Chunking with Transformers

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco
    pip install dm_control
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

