import os
import shutil
from subprocess import Popen, PIPE
import time
from utils import get_feature_meanings, parse_settings, DIRECTORY, ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY
from utils import fill_cox_tree, parse_cox_tree
import numpy as np
from math import exp
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd

# Replace paths if necessary!
EXEC_PATH = "streed2/STREED"
# EXEC_PATH = "../out/build/x64-Release/STREED.exe"
DATA_DIRECTORY = f"streed2/data/cox-survival-analysis"

TIME_OUT_IN_SECONDS = 600000000
PARETO_PRUNING = True

DATASET_TYPE_BIN = "binary"
DATASET_TYPE_NUM = "numeric"


# Takes a comma-separated dataset file and turns it into a file that STreeD can read
# Returns the list of feature names found in the first line of the file
#
# input_path    The path to the comma-separated file
# output_path   The path to the output file
def make_streed_compatible(input_path_bin, input_path_num, output_path):
    f_bin = open(input_path_bin)
    lines_bin = f_bin.read().strip().split("\n")
    f_bin.close()

    f_num = open(input_path_num)
    lines_num = f_num.read().strip().split("\n")
    f_num.close()

    feature_names = lines_bin[0].split(",")[2:]

    new_lines = []
    final_lines = []
    time = []
    for i in range(1, len(lines_bin)):
        # remove time and event
        time_i = float(lines_bin[i].split(",")[0])
        time.append(-time_i)
        final_lines.append("")
        bin_line = lines_bin[i].replace(",", " ")
        cnt = 0
        for j in range(len(bin_line)):
            if bin_line[j] == ' ':
                cnt += 1
                if cnt == 2:
                    bin_line = bin_line[j:]
                    break
        new_lines.append(lines_num[i].replace(",", " ") + bin_line)

    o = np.argsort(time, kind="mergesort")
    for i in range(len(o)):
        final_lines[i] = new_lines[o[i]]
    f = open(output_path, "w")
    f.write("\n".join(final_lines))
    f.close()

    cnt = lines_num[i].count(",")

    return feature_names, cnt - 1


# Runs STreeD with a set of parameters
# Returns resulting tree and the time needed to generate it (a negative time indicates a time out)
#
# parameters    The parameters to run the algorithm with
def run_streed(num_extra_cols, train_filename):
    args = []
    args.append("-cost-complexity")
    args.append(str(0))
    args.append("-max-depth")
    args.append(str(2))
    args.append("-max-num-nodes")
    args.append(str(3))
    args.append("-num-extra-cols")
    args.append(str(num_extra_cols))
    args.append("-min-leaf-node-size")
    args.append(str(5 * num_extra_cols))
    args.append("-survival-validation")
    args.append("log-like")
    args.append("-file")
    args.append(f"streed2/data/cox-survival-analysis/{train_filename}.txt")
    # Add addition arguments
    if TIME_OUT_IN_SECONDS > 0:
        args.extend(["-time", str(TIME_OUT_IN_SECONDS)])
    args.extend([
        "-task", "cox-survival-analysis"
    ])

    # Print executable call for convenience
    print(f"\033[35;1m{EXEC_PATH} {' '.join(args)}\033[0m")

    # Run executable
    proc = Popen([EXEC_PATH, *args], stdin=PIPE, stdout=PIPE)
    out, _ = proc.communicate()

    # Read the output from the console
    out_lines = [j.strip() for j in out.decode().split("\n")]
    time_line = [j for j in out_lines if j.startswith("CLOCKS FOR SOLVE:")][0]
    time = float(time_line.split()[-1])
    tree_line = [j for j in out_lines if j.startswith("Tree 0:")][0]
    tree = tree_line.split()[-1]

    return time, tree


def is_array(lst):
    return all(isinstance(x, (int, float)) for x in lst)


# Turn tree structure with numbers into tree structure with lambda's
#
# tree              The tree to convert
# feature_names     The names of the features, in order
# feature_meanings  The meanings of converted binary features
def serialize_tree_with_features(tree, feature_names, feature_meanings):
    if is_array(tree) == 1:
        # Leaf node
        return tree
    else:
        # Decision node
        # Try to get meaning of binary variable from the feature meanings map
        # If not found, the variable was already binary, so write a simple lambda for it
        feature, left, right = tree
        feature_name = feature_names[feature]
        feature_meaning = feature_meanings.get(feature_name, f"lambda x: x[\"{feature_name}\"]")

        # Serialize children and construct tree
        left_child = serialize_tree_with_features(left, feature_names, feature_meanings)
        right_child = serialize_tree_with_features(right, feature_names, feature_meanings)
        return f"[{feature_meaning},{left_child},{right_child}]"


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

    # plt.xlabel("Time (days)", fontsize=14, labelpad=0)
    # plt.ylabel("Survival rate", fontsize=14, labelpad=-10)
    # plt.yticks([0, 1], ["0", "1"])
    # plt.savefig(f"{DIRECTORY}/output/KM_distribution_{path}.svg", dpi=300)
    #
    # _, ax = plt.subplots(
    #     figsize=(2, 4),
    #     dpi=300,
    #     gridspec_kw=dict(left=0.12, right=0.99, bottom=0.11, top=0.9),
    #     facecolor="#FFFFFF"
    # )
    # ax.set_ylim([0, 1])
    #
    # for i in range(5):
    #     inst = tree.instances[i]
    #     s = 0
    #     for j in range(len(tree.coefs)):
    #         s += tree.coefs[j] * list(inst.feats.values())[j]
    #     s -= tree.model_offset
    #     distr = tree.breslow_distribution
    #     p2 = tree.get_expected_value(inst, inst)
    #     p3 = pow(distr(p2), exp(s))
    #     p4 = pow(distr(inst.time), exp(s))
    #     plt.scatter([p2], [p3], alpha=0.3)
    #     plt.scatter([inst.time], [p4], marker='^', alpha=0.3)
    #     plt.step(tree.unique_times, pow(tree.baseline_survival, exp(s)), alpha=0.2)
    # print(path)
    # print(tree.coefs)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#1a55FF', '#ffbb00', '#00d4ff', '#ff0044', '#004422',
              '#7200da', '#ffe600', '#00ffa1', '#ff55a3', '#008cff']
    t = 0
    for i in range(len(tree.instances)):
        if i % 20 != 0:
            continue
        t += 1
        inst = tree.instances[i]
        s = 0
        for j in range(len(tree.coefs)):
            s += tree.coefs[j] * list(inst.feats.values())[j]
        s -= tree.model_offset
        distr = tree.breslow_distribution
        ev = tree.get_expected_value(inst, inst)
        p1 = pow(distr(ev), exp(s))
        p2 = pow(distr(inst.time), exp(s))
        color = colors[t]
        plt.scatter([ev], [p1], color=color)
        plt.scatter([inst.time], [p2], marker='^', color=color)
        plt.step(tree.unique_times, pow(tree.baseline_survival, exp(s)), alpha=0.3, color=color)

    # f = open(f"{path}.txt", "w")
    # final_lines = []
    # for inst in tree.instances:
    #     strg = str(inst.time) + " " + str(inst.event)
    #     for j in range(len(tree.coefs)):
    #         strg = strg + " " + str(list(inst.feats.values())[j])
    #     final_lines.append(strg)
    # final_lines.reverse()
    # f.write("\n".join(final_lines))
    # f.close()
    plt.xlabel("Time (days)", fontsize=14, labelpad=0)
    plt.ylabel("Survival rate", fontsize=14, labelpad=-10)
    plt.yticks([0, 1], ["0", "1"])
    plt.savefig(f"{DIRECTORY}/output/joined_distribution_{path}.svg", dpi=300)
    plt.close()


def main():
    total_start_time = time.time()

    # Clear STreeD-datasets, these will be replaced anyway
    shutil.rmtree(DATA_DIRECTORY)
    for section in ["", "/train", "/test"]:
        os.mkdir(f"{DATA_DIRECTORY}{section}")

    dataset_directory_bin = f"{DIRECTORY}/datasets/{DATASET_TYPE_BIN}"
    dataset_directory_num = f"{DIRECTORY}/datasets/{DATASET_TYPE_NUM}"
    train_filename = "LeukSurv"
    train_path_bin = f"{dataset_directory_bin}/{train_filename}.txt"
    train_path_num = f"{dataset_directory_num}/{train_filename}.txt"
    streed_file = train_path_bin.replace(dataset_directory_bin, DATA_DIRECTORY)

    # Write STreeD files
    feature_names, num_extra_cols = make_streed_compatible(train_path_bin, train_path_num, streed_file)

    # Run STreeD
    # time_duration, tree = run_streed(num_extra_cols, train_filename)
    # if time_duration >= 0:
    #     print(f"\033[33;1m{tree}\033[0m")
    #     print(f"\033[34mTime: \033[1m{time_duration:.3f}\033[0;34m seconds\033[0m")
    tree = "[15,[50,[-0.520975,0.000000,0.022088,0.003427,0.019604,0.000000,0.018547],[-0.173628,0.678821,0.040158,0.002544,0.061661,0.000000,0.004338]],[44,[1.130926,0.960348,0.028138,0.003265,0.091126,-0.183788,-0.007964],[0.972430,1.521826,0.025237,0.003237,-0.064884,0.002366,-0.045905]]]"
    # Parse tree string to lambda-structure
    feature_meanings = get_feature_meanings(train_filename)
    tree = serialize_tree_with_features(eval(tree), feature_names, feature_meanings)
    print(tree)
    tree = eval(tree)
    tree = parse_cox_tree(tree)
    fill_cox_tree(tree, f"{ORIGINAL_DIRECTORY}/{train_filename}.txt", f"{NUMERIC_DIRECTORY}/{train_filename}.txt")
    print(tree)
    plot_leaf_distributions(tree)



if __name__ == "__main__":
    main()