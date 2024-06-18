import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
    # algorithms = ["streed", "ost", "ctree"]
    algorithms = ["streed", "coxstreedloglike", "coxstreedcindex"]
    # algorithms = ["streed", "ctree", "ost", "coxstreedloglike", "coxstreedcindex"]
    algorithm_name = {"ost": "OST", "streed": "SurTree", "ctree": "CTree", "coxstreedloglike": "CoxSurTreeLogLike", "coxstreedcindex": "CoxSurTreeCIndex"}
    # algorithm_name = {"streed": "SurTree", "ctree": "CTree"}
    for algorithm in algorithms:
        with open(f"{DIRECTORY}/output/{algorithm}_output.csv") as f:
            lines = f.read().strip().split("\n")
            for line in lines[1:]:
                _, settings, _results = [eval(j) for j in line.split(";")]
                if float(_results["runtime"]) < -1e-3 or float(_results["runtime"]) >= 600000000: continue
                results = {key: _results[key] for key in ["runtime", "num_nodes", "integrated_brier_score_ratio"]}
                results.update({key: settings[key] for key in ["max-depth","max-num-nodes", "hyper-tune", "cost-complexity"]})
                results["dataset"] = "_".join(settings["file"].replace("train/","").split("_"))
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
        dataset_info[dataset] = {"Number of instances": n_instances, "n_features": n_features, "n_binary_features": n_binary_features, "censoring": censoring}

        df.loc[df["dataset"] == dataset, "Number of instances"] = n_instances
        df.loc[df["dataset"] == dataset, "n_features"] = n_features
        df.loc[df["dataset"] == dataset, "censoring"] = censoring
        censoring_category = "Moderate"
        if censoring <= 0.25:
            censoring_category = "Low"
        elif censoring >= 0.75:
            censoring_category = "High"
        df.loc[df["dataset"] == dataset, "censoring_category"] = censoring_category

       
    df2 = df.melt(id_vars=["dataset", "method", "censoring_category", "Number of instances"], value_vars=["test_concordance_score", "integrated_brier_score_ratio"])
    
    HEIGHT = 2.2 # per-subplot height
    WIDTH = 7.0 # final width should be less than 6.95 inch
    N_COLS = 3
    N_ROWS = 2

    rel = sns.relplot(data=df2, x="Number of instances", y="value", 
                      hue='method', style='method', hue_order=["CoxSurTreeLogLike", "CoxSurTreeCIndex", "SurTree"], style_order=["CoxSurTreeLogLike", "CoxSurTreeCIndex", "SurTree"],
                      # hue='method', style='method', hue_order=["SurTree", "CTree"], style_order=["SurTree", "CTree"],
                      col='censoring_category', col_order=["Low", "Moderate", "High"],
                      row='variable', row_order=['test_concordance_score'],
                      kind='line',
                      height=HEIGHT, aspect = (WIDTH / (N_COLS *HEIGHT)),
                      facet_kws={"sharey":'row'})
    

    for (score, censoring_category), ax in rel.axes_dict.items():
        if score == "integrated_brier_score_ratio":
            ax.set_title("")
            ax.set_ylim(bottom = 0)
            if censoring_category == "Low":
                ax.set_ylabel("Integrated Brier Score")
            ax.set_xlabel("Number of instances")
        elif score == 'test_concordance_score':
            ax.set_title(f"Censoring: {censoring_category}")
            ax.set_ylim(bottom = 0.55)
            if censoring_category == 'Low':
                ax.set_ylabel("Harrell's C-index")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))    
    
    sns.move_legend(rel, "upper left", bbox_to_anchor=(0.1, 0.91), ncol=1, title="", frameon=True)

    plt.xscale('log')
    plt.xlim(1000, 10000)
    plt.subplots_adjust(wspace=0.1, hspace=0.15)

    #plt.show()
    plt.savefig(f"{DIRECTORY}/plots/synthetic_inc_n.pdf", bbox_inches="tight", pad_inches = 0)

    
if __name__ == "__main__":
    main()
