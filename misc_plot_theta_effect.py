import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import Instance, nelson_aalen
from utils import DIRECTORY


def plot(df):
    sns.set_context('paper')
    plt.rc('font', size=10, family='serif')
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('axes', labelsize='small', grid=False)
    plt.rc('legend', fontsize='x-small')
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('text', usetex=True)
    sns.set_palette("colorblind")

    # Creating the plot with Seaborn
    g = sns.relplot(data=df, x="x", y="y", col="theta", kind="line", legend=False,
                    height=1.6, aspect=(3.3 + 0.40) / (1.6 * 3))

    # Set the y-label for each subplot individually
    for ax in g.axes.flat:
        ax.set_ylabel("Survival probability")

    # Set other parameters and titles based on specific conditions
    for theta, ax in g.axes_dict.items():
        if theta == 0.5:
            ax.set_title("$\\theta < 1$")
        elif theta == 1.0:
            ax.set_title("$\\theta = 1$")
        else:
            ax.set_title("$\\theta > 1$")
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, 20.0)
        ax.set_xlabel("Time $\\rightarrow$")

    plt.subplots_adjust(wspace=0.1)
    plt.savefig(f"{DIRECTORY}/plots/theta_example.pdf", bbox_inches="tight", pad_inches=0)
    plt.close()

def shift_hazard_function(hazard_function, theta):
    xs = [0]
    ys = [1]
    prev_y = 1
    for x in np.arange(0, 21, .001):
        curr_y = np.exp(-theta * hazard_function(x))
        if curr_y != prev_y:
            xs.extend([x, x + 1e-6])
            ys.extend([prev_y, curr_y])
        prev_y = curr_y
    return pd.concat([pd.Series(xs, name="x"), pd.Series(ys, name="y"), pd.Series([theta] * len(xs), name="theta")], axis=1)

if __name__ == "__main__":
    instances = [Instance({"time": j, "event": 1}) for j in range(1, 21)]
    hazard_function = nelson_aalen(instances)

    dfs = []
    for theta in [0.3, 1, 3]:
        dfs.append(shift_hazard_function(hazard_function, theta))
    df = pd.concat(dfs)

    plot(df)

    print("\033[32;1mDone!\033[0m")
