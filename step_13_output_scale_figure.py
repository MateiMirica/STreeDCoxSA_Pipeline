import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gmean

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
    algorithms = ["streed", "ost", "ctree", "streed_nod2"]
    algorithm_name = {"ost": "OST", "streed": "SurTree", "streed_nod2": "SurTree no D2", "ctree": "CTree"}
    for algorithm in algorithms:
        with open(f"{DIRECTORY}/output/{algorithm}_output.csv") as f:
            lines = f.read().strip().split("\n")
            for line in lines[1:]:
                _, settings, _results = [eval(j) for j in line.split(";")]
                if _results["runtime"] >= 600000000 or _results["runtime"] < -50:
                    _results["runtime"] = 2000
                results = {key: _results[key] for key in ["runtime", "num_nodes", "integrated_brier_score_ratio"]}
                results.update({key: settings[key] for key in ["max-depth","max-num-nodes", "hyper-tune", "cost-complexity"]})
                results["dataset"] = settings["file"].replace("train/","")
                results["method"] = algorithm_name[algorithm]
                results.update({"train_"+key: _results["train"][key] for key in ["objective_score", "concordance_score"]})
                results.update({"test_"+key: _results["test"][key] for key in ["objective_score", "concordance_score"]})
                data.append(results)

    df = pd.DataFrame.from_dict(data)
    datasets = df["dataset"].unique()

    dataset_info = {}
    for dataset in datasets:
        with open(f"{DIRECTORY}/datasets/binary/{dataset}.txt") as f2:
            lines = f2.readlines()
            n_instances = len(lines) - 1
            n_binary_features = len(lines[0].split(",")) - 2
            censoring = 1.0 - sum([int(line.split(",")[1]) for line in lines[1:]]) / n_instances
        with open(f"{DIRECTORY}/datasets/original/{dataset}.txt") as f2:
            lines = f2.readlines()
            n_features = len(lines[0].split(",")) - 2
        dataset_info[dataset] = {"n_instances": n_instances, "n_features": n_features, "n_binary_features": n_binary_features, "censoring": censoring}

        df.loc[df["dataset"] == dataset, "n_instances"] = n_instances
        df.loc[df["dataset"] == dataset, "n_features"] = n_features
        df.loc[df["dataset"] == dataset, "censoring"] = censoring
        censoring_category = "Moderate"
        if censoring <= 0.25:
            censoring_category = "Low"
        elif censoring >= 0.75:
            censoring_category = "High"
        df.loc[df["dataset"] == dataset, "censoring_category"] = censoring_category


    df["Features"] = df[["n_features"]].apply(lambda row: f"f={int(row['n_features']/6):d}", axis=1)
    df["Method"] = df["method"]

    plt.figure(figsize=(3.3+0.3, 1.7))

    g = sns.lineplot(data = df, x="max-depth", y='runtime',
                 hue="Method", hue_order=["SurTree", "SurTree no D2", "OST", "CTree"],
                 style='Features', style_order=["f=1", "f=2"]
                )
    g.set_yscale("log")
    g.set_xlim(2, 5)
    g.set_ylim(0.1, 600000000)
    g.set_ylabel("Run time (s)")
    g.set_xlabel("Maximum depth")

    plt.subplots_adjust(right=0.6)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), ncol=1, title="", frameon=False)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{DIRECTORY}/plots/runtime_d.pdf", bbox_inches="tight", pad_inches = 0)


    # remove time-outs
    rts = df.groupby(["dataset", "method", "max-depth", "Features"])["runtime"].mean().unstack("method")
    instances_within_time_out_ix = np.column_stack([rts[m] <= 600000000 for m in ["SurTree", "SurTree no D2"]]).all(axis=1)
    instances_within_time_out = pd.Series(instances_within_time_out_ix, index=rts.index)
    df = df[df.apply(lambda x: instances_within_time_out.loc[x["dataset"], x["max-depth"], x["Features"]], axis=1)]


    gr = df.groupby(["dataset", "method", "max-depth", "Features"])["runtime"].mean().unstack("method")

    print("Gain from special d2-solver: ", gmean(gr["SurTree no D2"] / gr["SurTree"]))

if __name__ == "__main__":
    main()

