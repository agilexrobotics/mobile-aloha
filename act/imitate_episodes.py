import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
    set_seed(1)   

    # command line parameters
    is_eval = args['eval']                    # 否执行评估操作 
    ckpt_dir = args['ckpt_dir']               # 检查点文件的保存目录
    policy_class = args['policy_class']       # 策略类别，首字母大写
    onscreen_render = args['onscreen_render'] # 是否进行屏幕渲染
    task_name = args['task_name']             # 任务名称
    batch_size_train = args['batch_size']     # 训练批处理大小
    batch_size_val = args['batch_size']       # validation验证集
    num_epochs = args['num_epochs']           # 训练周期

    # get task parameters # 是否是仿真数据
    is_sim = task_name[:4] == 'sim_'
    
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    
    # 不是仿真数据就加载aloha_scripts.constants的配置文件
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14             # 状态维度
    lr_backbone = 1e-5         # 学习率
    backbone = 'resnet18'      # 特征提取层
    # 如果是ACT策略
    if policy_class == 'ACT':
        enc_layers = 4            # 编码层
        dec_layers = 7            # 解码层
        nheads = 8                # 多头注意力头数
       
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }
 
    # 如果评估   这里是加载的best权重, 只有训练后才能加--eval
    if is_eval:
        ckpt_names = [f'policy_best.ckpt']  # 权重路径
        results = []
        
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
       

    # 数据集加载  路径 回合数 相机名字 训练集批处理大小 验证集批处理大小
    # 返回处理好的训练集和验证集 还有数据集均值和方差
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats  保存数据集的均值和方差
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 这里会创建训练模型  训练 每100周期权重，最后一次和损失最低的权重， 返回了损失最低是状态
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    
    # 周期数, 最低的评价损失值, 权重状态字典
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint  保存权重
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)  
    
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    policy = make_policy(policy_class, policy_config)
    
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    
    # 加载action_mean, action_std, qpos_mean, qpos_std 14维
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        # from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        
        # 创建2个臂
        env = make_real_env(init_node=True)
        env_max_reward = 0
    
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    # 展示
    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    
    for rollout_id in range(num_rollouts):
        rollout_id += 0               # +0 就是从0-50, 如果加5就是 5-54
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        
        # 2个从臂的重启 
        ts = env.reset()  # 初始化  初始时刻从臂的状态

        ### onscreen render  
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:    # 1000, 1000+100, 14
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        # 1 1000轮 状态14维
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        
        with torch.inference_mode():  # 模型处于推断模式
            
            # 每个回合的步数
            for t in range(max_timesteps):
                
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation    
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                
                # 获取从臂的动作  obs都是从臂的
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        
                        # 推理(从臂的q)  推出的主臂的action
                        all_actions = policy(qpos, curr_image)
                    
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        
                        actions_for_curr_step = all_time_actions[:, t]

                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                
                # 后处理到raw_action
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)   # 归一化处理 均值和方差
                target_qpos = action

                ### step the environment
                # 获得当前的主臂的动作后, 在推理
                ts = env.step(target_qpos)   # 更新ts
                
                # 根据主臂的动作, 从臂跟随  因为有话题，obs观测值就更新了
                # import dm_env, collections
                # obs = collections.OrderedDict()  # 有序的字典
        
                # qpos qvel effort 14维度 base_vel 2维度
                # obs['qpos'] = np.random.rand(14)
                # obs['qvel'] = np.random.rand(14)
                # obs['effort'] = np.random.rand(14)
                # # obs['images'] = image_dict
                # obs['base_vel'] = np.array([0.1, 0.3])
                

                # dm_env.TimeStep(
                #     step_type=dm_env.StepType.MID,
                #     reward=None,
                #     discount=None,
                #     observation=obs)


                ### for visualization
                qpos_list.append(qpos_numpy)          # 从臂的终止
                target_qpos_list.append(target_qpos)  # 从臂的q
                rewards.append(ts.reward)             # 从臂的奖励
            plt.close()
        
        if real_robot:
            
            print(env.puppet_bot_left, env.puppet_bot_right)
        
            # move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    # 图像转到cuda
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)  # 设置随机种子

    # 根据规则定义模型 ACT为例子 build_ACT_model_and_optimizer
    
    policy = make_policy(policy_class, policy_config)
    # 打印模型
    # print(policy)
    
    # qpos: batch, qpos_dim
    # image: batch, num_cam, channel, height, width
    # env_state: None                                   没有值
    # actions: batch, seq, action_dim   
    # from torchsummary import summary
    # summary(policy, (3, 224, 224)) 
    
    # exit(-1)
    
    
    
    policy.cuda()  

    # 优化器配置 optimizer = torch.optim.AdamW
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    # 多少个周期
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # 验证 validation
        with torch.inference_mode():
            policy.eval()  # 模型处理评估模式。
            # 模型会使用保存的移动平均值（Batch Normalization）
            # 和不用 Dropout
            # 不更新梯度
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)   # 推理
                epoch_dicts.append(forward_dict)            # 保存推理结果
            
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            
            if epoch_val_loss < min_val_loss:               # 保存损失最小时的权重
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
        print(f'Val loss:   {epoch_val_loss:.5f}')
        
        summary_string = ''
        
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # 训练 training
        policy.train()   # 模型处理训练撞人
        
        optimizer.zero_grad() # 每个训练步骤开始时将模型参数的梯度归   # 清零梯度
        
        print("len(train_dataloader): ", len(train_dataloader), "\n")
        
        for batch_idx, data in enumerate(train_dataloader):
            
            # 反向传播 就是下面3行
            # forward_dict = forward_pass(data, policy)
            
            # 3个相机, pose是2*14 
            image_data, qpos_data, action_data, is_pad = data
            
            # 数据转cuda设备上
            image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
            

            # print(qpos_data.shape, image_data.shape, action_data.shape, is_pad.shape)

            # torch.Size([8, 14]) torch.Size([8, 3, 3, 480, 640]) 
            # torch.Size([8, 300, 14]) torch.Size([8, 300])
            # 模型前向传播并计算损失   这里调用__call__函数 使用类ACTPolicy实例化policy
            # ACTPolicy类中 创建了model 调用policy()时会调用__call__, __call__函数中会调用model
            forward_dict = policy(qpos_data, image_data, action_data, is_pad) # TODO remove None
            
            from torchsummary import summary
            # summary(policy.model, [(qpos_data,), (image_data,),(action_data,), (is_pad,)])


            ''' 
             def __call__ 函数计算损失  
             pose 
             image 
             actions
             is_pad:   [b, 300]
             
                def __call__(self, qpos, image, actions=None, is_pad=None):
                    env_state = None
                    
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                    
                    image = normalize(image)  # 图像归一化
                    
                    if actions is not None: # training time
                        actions = actions[:, :self.model.num_queries]
                        is_pad = is_pad[:, :self.model.num_queries]

                        a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
                        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                        
                        loss_dict = dict()

                        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                        
                        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                        
                        loss_dict['l1'] = l1
                        loss_dict['kl'] = total_kld[0]
                        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
                        
                        return loss_dict
                    else: # inference time
                        a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
                        return a_hat
            '''


            
            # backward
            loss = forward_dict['loss']
            
            loss.backward()   # 反向传播
            optimizer.step()  # 优化步骤
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))   # 记录训练过程
        
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        
        print(f'Train loss: {epoch_train_loss:.5f}')
        
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        #每100个周期保存一次权重
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # 保存最后一次权重
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    # 保存损失最低的一次权重
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        print(key)
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        
        val_values = [summary[key].item() for summary in validation_history]
        
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    
    # query的数量
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    
    # 隐藏层的维度
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
