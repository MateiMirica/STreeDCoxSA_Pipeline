import pandas as pd
import os
from contextlib import redirect_stdout
from scipy.stats import gmean

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

if not os.path.exists(f"{DIRECTORY}/tables"):
    os.mkdir(f"{DIRECTORY}/tables")

def main():
    # Load data for each algorithm
    data = []
    algorithms = ["ctree", "ost", "streed", "coxstreedloglike", "coxstreedcindex"]
    for algorithm in algorithms:
        with open(f"{DIRECTORY}/output/{algorithm}_output.csv") as f:
            lines = f.read().strip().split("\n")
            for line in lines[1:]:
                _, settings, _results = [eval(j) for j in line.split(";")]
                results = {key: _results[key] for key in ["runtime", "num_nodes", "integrated_brier_score_ratio"]}
                results.update({key: settings[key] for key in ["max-depth","max-num-nodes", "hyper-tune", "cost-complexity"]})
                results["dataset"] = settings["file"].replace("train/","").split("_")[0]
                results["method"] = algorithm
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


    timeouts = df["runtime"] >= 600000000
    df.loc[timeouts, "test_concordance_score"] = 0.5
    df.loc[timeouts, "integrated_brier_score_ratio"] = 0.0

    means_ibs = df.groupby(["dataset", "method"]).mean(numeric_only=True)["integrated_brier_score_ratio"].unstack("method")
    rank_ibs = means_ibs.round(2).rank(axis=1, ascending=False).mean(axis=0)
    best_ibs = means_ibs.max(axis=1)
    means_hc = df.groupby(["dataset", "method"]).mean(numeric_only=True)["test_concordance_score"].unstack("method")
    best_hc = means_hc.max(axis=1)
    rank_hc = means_hc.round(2).rank(axis=1, ascending=False).mean(axis=0)
    means_runtime = df.groupby(["dataset", "method"]).mean(numeric_only=True)["runtime"].unstack("method")

    print(means_runtime)

    wins = {t: {a: 0 for a in algorithms} for t in ["ibs", "hc"]}

    with open(f"{DIRECTORY}/tables/hc_ibs_survset.txt", "w") as f:
        with redirect_stdout(f):
            for dataset in sorted(datasets):
                print("{} &".format(dataset.capitalize()))
                print(f"{dataset_info[dataset]['n_instances']} & {dataset_info[dataset]['censoring']*100:.1f}\% & {dataset_info[dataset]['n_features']} & {dataset_info[dataset]['n_binary_features']} &")

                sep = "&"
                for algorithm in algorithms:
                    score = means_hc.loc[dataset, algorithm]
                    score_str = f"{score:.2f}"
                    
                    if means_runtime.loc[dataset, algorithm] >= 600000000:
                        print(f"-  {sep} % Timeout {algorithm}, HC {score_str}")
                        continue

                    if round(score, 2)  >= round(best_hc[dataset], 2):
                        score_str = "\\textbf{" + score_str + "}"
                        wins["hc"][algorithm] += 1
                    print(f"{score_str} {sep} % HC {algorithm}")

                for algorithm in algorithms:
                    sep = "\\\\" if algorithm == algorithms[-1] else "&"
                    score = means_ibs.loc[dataset, algorithm]
                    score_str = f"{score:.2f}"

                    if means_runtime.loc[dataset, algorithm] >= 600000000:
                        print(f"-  {sep} % Timeout {algorithm}, IBS {score_str}")
                        continue

                    if round(score, 2) >= round(best_ibs[dataset], 2):
                        score_str = "\\textbf{" + score_str + "}"
                        wins["ibs"][algorithm] += 1
                    print(f"{score_str} {sep} % IBS {algorithm}")

            print("\midrule\nWins per metric & &&&&")
            sep = "&"
            for algorithm in algorithms:
                wins_str = f"{wins['hc'][algorithm]}"
                if wins['hc'][algorithm] >= max(wins['hc'].values()):
                    wins_str = "\\textbf{" + wins_str + "}"
                print(f"{wins_str} {sep} % {algorithm} # HC wins")

            for algorithm in algorithms:
                sep = "\\\\" if algorithm == algorithms[-1] else "&"
                wins_str = f"{wins['ibs'][algorithm]}"
                if wins['ibs'][algorithm] >= max(wins['ibs'].values()):
                    wins_str = "\\textbf{" + wins_str + "}"
                print(f"{wins_str} {sep} % {algorithm} # IBS wins")

            print("Average rank & &&&&")
            sep = "&"
            for algorithm in algorithms:
                rank_str = f"{rank_hc[algorithm]:.2f}"
                if rank_hc[algorithm] <= min(rank_hc):
                    rank_str = "\\textbf{" + rank_str + "}"
                print(f"{rank_str} {sep} % {algorithm} # HC rank")

            for algorithm in algorithms:
                sep = "\\\\" if algorithm == algorithms[-1] else "&"
                rank_str = f"{rank_ibs[algorithm]:.2f}"
                if rank_ibs[algorithm] <= min(rank_ibs):
                    rank_str = "\\textbf{" + rank_str + "}"
                print(f"{rank_str} {sep} % {algorithm} # IBS rank")
            
                    
    gr = df.groupby(["dataset", "method"])["runtime"].mean().unstack("method")

    # print("Times faster, OST vs SurTree: ", gmean((gr["ost"] / gr["streed"])))



if __name__ == "__main__":
    main()
