import pickle
import numpy as np
import matplotlib
matplotlib.rc('font', size=14)
matplotlib.rcParams['figure.figsize'] = (5, 3)
matplotlib.rcParams["legend.markerscale"] = 2
matplotlib.rcParams["legend.columnspacing"] = 1.5
matplotlib.rcParams["legend.labelspacing"] = 0.4
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker



def plot_accumulate_reward_dynamics(zero_reward_list, non_zero_reward_list, zero_label, non_zero_label, env_name, alg_name, file_path, step_size):

    zero_reward_list = np.mean(zero_reward_list, axis=0)
    non_zero_reward_list = np.mean(non_zero_reward_list, axis=0)
    
    plt.plot(zero_reward_list, label=zero_label)
    plt.plot(non_zero_reward_list, label=non_zero_label)

    plt.xlabel("training steps")
    plt.ylabel("total reward")

    plt.legend(loc=2)
    plt.tight_layout(pad=0)

    plt.xlabel("step")
    plt.ylabel("accumulated reward")
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step_size))
    
    plt.savefig(file_path + "/" + env_name + "_" + alg_name + "_reward_fig.png")
    plt.close()


def plot_step_count(zero_count_list, non_zero_count_list, zero_label, non_zero_label, env_name, alg_name, file_path, step_size):
    
    zero_count_list = np.mean(zero_count_list, axis=0)
    non_zero_count_list = np.mean(non_zero_count_list, axis=0)
    
    plt.plot(zero_count_list, label=zero_label)
    plt.plot(non_zero_count_list, label=non_zero_label)

    plt.xlabel("training steps")
    plt.ylabel("count")

    plt.legend(loc=2)
    plt.tight_layout(pad=0)

    plt.xlabel("step")
    plt.ylabel("trajectory length")
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step_size))
    
    plt.savefig(file_path + "/" + env_name + "_" + alg_name + "_count_fig.png")
    plt.close()


def main(alg_name, env_name, file_path):
    result = pickle.load(open("./pickle_data/" + env_name + "_" + alg_name + "_result.pl", "rb"))
    zero_reward_list = result["zero_reward_list"]
    zero_count_list = result["zero_count_list"]
    non_zero_reward_list = result["non_zero_reward_list"]
    non_zero_count_list = result["non_zero_count_list"]
    step_size = 1000

    if env_name == "cart_pole":
        zero_label = r"zero"
        # non_zero_label = r"$\frac{10}{9(1-\gamma)}$"
        non_zero_label = r"non-zero"

        plot_step_count(zero_count_list=zero_count_list, non_zero_count_list=non_zero_count_list, 
                        zero_label=zero_label, non_zero_label=non_zero_label, env_name=env_name, 
                        alg_name=alg_name, file_path=file_path, step_size=step_size)
        plot_accumulate_reward_dynamics(zero_reward_list=zero_reward_list, non_zero_reward_list=non_zero_reward_list, 
                        zero_label=zero_label, non_zero_label=non_zero_label, env_name=env_name, alg_name=alg_name, 
                        file_path=file_path, step_size=step_size)

    elif env_name == "mountain_car":
        zero_label = r"zero"
        # non_zero_label = r"$-\frac{10}{9(1-\gamma)}$"
        non_zero_label = r"non-zero"

        if alg_name in ["ppo", "a2c"]:
            step_size = 2000

        plot_step_count(zero_count_list=zero_count_list, non_zero_count_list=non_zero_count_list, 
                        zero_label=zero_label, non_zero_label=non_zero_label, env_name=env_name, 
                        alg_name=alg_name, file_path=file_path, step_size=step_size)
        plot_accumulate_reward_dynamics(zero_reward_list=zero_reward_list, non_zero_reward_list=non_zero_reward_list, 
                        zero_label=zero_label, non_zero_label=non_zero_label, env_name=env_name, alg_name=alg_name, 
                        file_path=file_path, step_size=step_size)

    elif env_name == "stochatsic_positive_mdp":
        zero_label = r"zero"
        # non_zero_label = r"$\frac{ 10(r_{\min} - r_{\max}) }{ 9 \delta \gamma^{|S|-1} (1-\gamma) }$"
        non_zero_label = r"non-zero"

        if alg_name in ["a2c"]:
            step_size = 4000

        plot_step_count(zero_count_list=zero_count_list, non_zero_count_list=non_zero_count_list, 
                        zero_label=zero_label, non_zero_label=non_zero_label, env_name=env_name, 
                        alg_name=alg_name, file_path=file_path, step_size=step_size)
        plot_accumulate_reward_dynamics(zero_reward_list=zero_reward_list, non_zero_reward_list=non_zero_reward_list, 
                        zero_label=zero_label, non_zero_label=non_zero_label, env_name=env_name, alg_name=alg_name, 
                        file_path=file_path, step_size=step_size)

    elif env_name == "stochatsic_negative_mdp":
        zero_label = r"zero"
        # non_zero_label = r"$\frac{10 (r_{\max} - r_{\min}) }{ 9\delta \gamma^{|S|-1} (1-\gamma) }$"
        non_zero_label = r"non-zero"

        plot_step_count(zero_count_list=zero_count_list, non_zero_count_list=non_zero_count_list, 
                        zero_label=zero_label, non_zero_label=non_zero_label, env_name=env_name, 
                        alg_name=alg_name, file_path=file_path, step_size=step_size)
        plot_accumulate_reward_dynamics(zero_reward_list=zero_reward_list, non_zero_reward_list=non_zero_reward_list, 
                        zero_label=zero_label, non_zero_label=non_zero_label, env_name=env_name, alg_name=alg_name, 
                        file_path=file_path, step_size=step_size)

    

    
if __name__ == "__main__":
    result1 = pickle.load(open("./pickle_data/stochatsic_positive_mdp_dqn_result1.pl", "rb"))
    result2 = pickle.load(open("./pickle_data/stochatsic_positive_mdp_dqn_result3.pl", "rb"))

    pickle.dump(
                {
                    "zero_reward_list": result1["zero_reward_list"] + result2["zero_reward_list"],
                    "zero_count_list": result1["zero_count_list"] + result2["zero_count_list"],
                    "non_zero_reward_list": result1["non_zero_reward_list"] + result2["non_zero_reward_list"],
                    "non_zero_count_list": result1["non_zero_count_list"] + result2["non_zero_count_list"],
                    "param_dict": result1["param_dict"]
                },
                open("./pickle_data/stochatsic_positive_mdp_dqn_result.pl", "wb")
            )

    FILE_PATH = "./img"
    for ALG_NAME in ["dqn", "ppo", "a2c"]:
        for ENV_NAME in ["cart_pole", "mountain_car", "stochatsic_positive_mdp", "stochatsic_negative_mdp"]:
            main(alg_name=ALG_NAME, env_name=ENV_NAME, file_path=FILE_PATH)
