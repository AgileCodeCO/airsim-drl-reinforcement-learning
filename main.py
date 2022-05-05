import os
import glob
import time
from datetime import datetime
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

import torch
import numpy as np

from agents.PPO import PPO
from environment.drl_environment import DRLEnvironment

def train():
   
    env_name = "DRL"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 30                   # max timesteps in one episode
    max_training_timesteps = 20000   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 2      # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2         # log avg reward in the interval (in num timesteps)
    save_model_freq = 200          # save model frequency (in num timesteps)

    action_std = 0.60                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = 1000  # action_std decay frequency (in num timesteps)
    
    update_timestep = max_ep_len * 2      # update policy every n timesteps
    K_epochs = 40               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # Higher discount factor for continuos actions

    lr_actor = 0.0001       # learning rate for actor network
    lr_critic = 0.0001       # learning rate for critic network

    random_seed = 47         # set random seed if required (0 = no random seed)
   
    env = DRLEnvironment(viz_image_cv2=False, observation_type="lidar")

    # state space dimension
    state_dim = env.get_observation_space()[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space[0]
    else:
        action_dim = env.action_space[0]

    
    log_dir = "logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    run_num_pretrained = 7      #### change this to prevent overwriting weights in same env_name folder
    continue_training = False

    directory = "models"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("Archivo de pesos : " + checkpoint_path)
   
    if random_seed:        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    
    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    if continue_training:
        print("Cargando modelo anterior : " + checkpoint_path)
        ppo_agent.load(checkpoint_path)

    env.init_race_environment()
    # training loop
    while time_step <= max_training_timesteps:

        state = env.start_race()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episodio : {} \t\t Timestep : {} \t\t Recompensa : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:               
                ppo_agent.save(checkpoint_path)
                print("Modelo guardado")
                print("Tiempo de entrenamiento : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        
        env.reset()               
        time.sleep(3)

    log_f.close()

def test():
   
    env_name = "DRL"
    has_continuous_action_space = True
    max_ep_len = 300           # max timesteps in one episode
    action_std = 0.10          # set same std for action distribution which was used while saving

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.001           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    env = DRLEnvironment(viz_image_cv2=False, observation_type="lidar")

    # state space dimension
    state_dim = env.get_observation_space()[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space[0]
    else:
        action_dim = env.action_space[0]

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 47          #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 5      #### set this to load a particular checkpoint num

    directory = "models" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("Cargando modelo anterior : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    
    env.init_race_environment()

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.start_race()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            ep_reward += reward

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episodio: {} \t\t Recompensa: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
        
        env.reset()               
        time.sleep(3)


    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("Recompensa promedio : " + str(avg_test_reward))

    print("============================================================================================")
    
def plot():
   
    env_name = 'DRL'

    fig_num = 0     #### change this to prevent overwriting figures in same env_name folder
    plot_avg = False    # plot average of all runs; else plot all runs separately
    fig_width = 10
    fig_height = 6

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson','gray', 'black']

    # make directory for saving figures
    figures_dir = "plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + env_name + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + '/PPO_' + env_name + '_fig_' + str(fig_num) + '.png'

    # get number of log files in directory
    log_dir = "logs_to_plot" + '/' + env_name + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    for run_num in range(num_runs):

        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
       
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)

        all_runs.append(data)

    ax = plt.gca()

    if plot_avg:
        # average all runs
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg['reward_smooth'] = data_avg['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

        data_avg.plot(kind='line', x='timestep' , y='reward_smooth',ax=ax,color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='timestep' , y='reward_var',ax=ax,color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)

        # keep only reward_smooth in the legend and rename it
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0]], ["reward_avg_" + str(len(all_runs)) + "_runs"], loc=2).remove()

    else:
        for i, run in enumerate(all_runs):
            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            run['reward_smooth_' + str(i)] = run['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            # plot the lines
            run.plot(kind='line', x='timestep' , y='reward_smooth_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='timestep' , y='reward_var_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)

        # keep alternate elements (reward_smooth_i) in the legend
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for i in range(len(handles)):
            if(i%2 == 0):
                new_handles.append(handles[i])
                new_labels.append(labels[i])
        ax.legend(new_handles, new_labels, loc=2).remove()

    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)

    plt.title(env_name, fontsize=14)
    
    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)

    plt.savefig(fig_save_path)
    
    plt.show()

def main(args):
    if args.mode == 'train':
        train()
         
    if args.mode == 'test':
        test()
        
    if args.mode == 'plot':
        plot()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "test",
            "plot"
        ],
        default="train",
    )
    
    args = parser.parse_args()
    main(args)