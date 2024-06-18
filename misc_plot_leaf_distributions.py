from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd
from utils import fill_tree, parse_tree
from utils import DIRECTORY

def plot_leaf_distributions(tree, path=""):
    if tree.trees:
        plot_leaf_distributions(tree.trees[1], path + "V")
        plot_leaf_distributions(tree.trees[0], path + "X")
        return
    
    plt.clf()
    _, ax = plt.subplots(
        figsize=(2, 4),
        dpi=300,
        gridspec_kw=dict(left=0.12, right=0.99, bottom=0.11, top=0.9),
        facecolor="#FFFFFF"
    )
    ax.set_ylim([0, 1])

    df = pd.DataFrame.from_dict({
        "time": [inst.time for inst in tree.instances],
        "event": [inst.event for inst in tree.instances],
    })

    kmf = KaplanMeierFitter(label="")
    kmf.fit(df["time"], df["event"])
    kmf.plot(color="#DF1F1F", ci_show=False, linewidth=3)
    ax.get_legend().remove()

    plt.xlabel("Time (days)", fontsize=14, labelpad=0)
    plt.ylabel("Survival rate", fontsize=14, labelpad=-10)
    plt.yticks([0, 1], ["0", "1"])
    plt.savefig(f"{DIRECTORY}/output/distribution_{path}.svg", dpi=300)
    plt.close()

if __name__ == "__main__":
    # IMPORTANT: If LeukSurv.txt does not exist:
    #   - Remove the "n >= 2000"-requirement in step_1_download_datasets.py to make sure it is imported, or;
    #   - Choose a different dataset and tree to plot.
    tree = parse_tree([lambda x: x['num_age'] > 57.0,[lambda x: x['num_wbc'] > 38.65,[0.499864],[0.883823]],[lambda x: x['num_age'] > 71.0,[1.137998],[1.814392]]])
    fill_tree(tree, f"{DIRECTORY}/datasets/original/LeukSurv.txt")

    plot_leaf_distributions(tree)


    print("\033[32;1mDone!\033[0m")
