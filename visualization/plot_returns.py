import matplotlib
matplotlib.rc('font', size=14)
matplotlib.rcParams['figure.figsize'] = (5, 4)
matplotlib.rcParams["legend.markerscale"] = 2
matplotlib.rcParams["legend.columnspacing"] = 1.5
matplotlib.rcParams["legend.labelspacing"] = 0.4
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker



def plt_accumulate_reward_dynamics(zero_reward_list, non_zero_reward_list, env_name, alg_name, file_path):

    total_df = pd.DataFrame()
    for i in range(len(zero_reward_list)):
        data_dict_list = []
        for j in range(len(zero_reward_list[i])):
            tmp_dict = {}
            tmp_dict["step"] = j
            tmp_dict["zero"] = zero_reward_list[i][j]
            tmp_dict["non-zero"] = non_zero_reward_list[i][j]
            # tmp_dict["seed"] = i
            data_dict_list.append(tmp_dict)
        df = pd.DataFrame().from_dict(data_dict_list)
        total_df = pd.concat([total_df, df], ignore_index=True)
    
    l1 = sns.lineplot(data=total_df, x="step", y="zero", legend='brief', label="Zero")
    l2 = sns.lineplot(data=total_df, x="step", y="non-zero", legend='brief', label="Non-zero")
    
    plt.legend(loc=0)
    plt.tight_layout(pad=0)

    plt.xlabel("step")
    plt.ylabel("accumulated reward")
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    
    plt.savefig(file_path + "/" + env_name + "_" + alg_name + "_reward_fig.png")
    plt.close()

