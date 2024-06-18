import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

if not os.path.exists(f"{DIRECTORY}/plots"):
    os.mkdir(f"{DIRECTORY}/plots")

def main():

    sns.set_context('paper')
    plt.rc('font', size=10, family='serif')
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('axes', labelsize='small', grid=True)
    plt.rc('legend', fontsize='x-small')
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)
    plt.rc('text', usetex = True)
    sns.set_palette("colorblind")

    data = []
    algorithms = ["streed", "ost", "ctree"]
    algorithm_name = {"ost": "OST", "streed": "SurTree", "ctree": "CTree"}
    for algorithm in algorithms:
        with open(f"{DIRECTORY}/output/{algorithm}_output.csv") as f:
            lines = f.read().strip().split("\n")
            for line in lines[1:]:
                _, settings, _results = [eval(j) for j in line.split(";")]
                if _results["runtime"] >= 600000000 or _results["runtime"] < -1e6:
                    _results["runtime"] = 1200
                results = {key: _results[key] for key in ["runtime", "num_nodes", "integrated_brier_score_ratio"]}
                results.update({key: settings[key] for key in ["max-depth","max-num-nodes", "hyper-tune", "cost-complexity"]})
                results["dataset"] = settings["file"].replace("train/","")
                results["method"] = algorithm_name[algorithm]
                results.update({"train_"+key: _results["train"][key] for key in ["objective_score", "concordance_score"]})
                results.update({"test_"+key: _results["test"][key] for key in ["objective_score", "concordance_score"]})
                data.append(results)

    df = pd.DataFrame.from_dict(data)
    datasets = df["dataset"].unique()
    _means =  df.groupby(["method", "max-depth"])["train_objective_score"].mean()
    means = df.groupby(["method", "max-depth"])[["train_objective_score"]].mean().reset_index()

    print(means)

    for d in range(2,6):
        sur = _means["SurTree", d]
        ost = _means[("OST", d)]
        diff = ((sur-ost) / ost) * 100
        print(f"D={d}, SurTree is {diff:.2f}% better than OST.")
    for d in range(2,6):
        sur = _means["SurTree", d]
        ctree = _means[("CTree", d)]
        diff = ((sur-ctree) / ctree) * 100
        print(f"D={d}, SurTree is {diff:.2f}% better than CTree.")
        
    plt.figure(figsize=(3.3+0.3, 1.6))

    method_order=["SurTree", "OST", "CTree"]
    g = sns.lineplot(data = means, x="max-depth", y='train_objective_score',
                 hue="method", hue_order=method_order,
                 style='method', style_order=method_order
                )
    g.set_xlim(2, 5)
    #g.set_ylim(0.1, 0.25)
    g.set_ylabel("Training score")
    g.set_xlabel("Maximum depth")
    g.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.subplots_adjust(right=0.6)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), ncol=1, title="", frameon=False)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{DIRECTORY}/plots/train_score.pdf", bbox_inches="tight", pad_inches = 0)

if __name__ == "__main__":
    main()

