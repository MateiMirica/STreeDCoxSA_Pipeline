import os
import shutil
from subprocess import Popen, PIPE
import time
from utils import get_feature_meanings, parse_settings, DIRECTORY
import numpy as np

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


pareto_front = []


def create_pareto_key(parameters):
    filename = parameters["core-file"]
    depth = parameters["max-depth"]

    if filename.startswith("generated_dataset_"):
        n, f, c = [int(j) for j in filename.split("_")[2:5]]
        return (True, n, f, c, depth)
    else:
        return (False, filename, depth, None, None)


def contains_pareto_key(key):
    if key[0]:
        _, n, f, c, depth = key
        for is_synthetic, other_n, other_f, other_c, other_depth in pareto_front:
            if is_synthetic and other_n <= n and other_f <= f and other_c == c and other_depth <= depth:
                return True
        return False
    else:
        _, filename, depth, _, _ = key
        for is_synthetic, other_filename, other_depth, _, _ in pareto_front:
            if not is_synthetic and other_filename == filename and other_depth <= depth:
                return True
        return False


# Runs STreeD with a set of parameters
# Returns resulting tree and the time needed to generate it (a negative time indicates a time out)
#
# parameters    The parameters to run the algorithm with
def run_streed(parameters, num_extra_cols):
    global pareto_front
    pareto_key = create_pareto_key(parameters)
    if PARETO_PRUNING:
        if contains_pareto_key(pareto_key):
            return -1e9, "[None]"

    # Convert parameters to arguments
    args = []
    for key, value in parameters.items():
        if key == "core-file" or key == 'test-file' or key == 'max-depth' or key == 'max-num-nodes':
            continue
        args.append(f"-{key}")
        args.append(f"{value}")
    args.append("-max-depth")
    args.append(str(2))
    args.append("-max-num-nodes")
    args.append(str(3))
    args.append("-num-extra-cols")
    args.append(str(num_extra_cols))
    args.append("-survival-validation")
    args.append("log-like")
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

    # Check whether there was a time-out first
    if "No tree found" in out_lines:
        if PARETO_PRUNING:
            pareto_front.append(pareto_key)
        return -time, "[None]"

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


def main():
    total_start_time = time.time()

    # Clear STreeD-datasets, these will be replaced anyway
    shutil.rmtree(DATA_DIRECTORY)
    for section in ["", "/train", "/test"]:
        os.mkdir(f"{DATA_DIRECTORY}{section}")

    # Read settings
    params_settings = parse_settings(f"{DIRECTORY}/output/settings.txt")

    results = []
    try:
        dataset_directory_bin = f"{DIRECTORY}/datasets/{DATASET_TYPE_BIN}"
        dataset_directory_num = f"{DIRECTORY}/datasets/{DATASET_TYPE_NUM}"

        for params in params_settings:
            # Change files to new STreeD files
            train_filename = params["file"]
            train_path_bin = f"{dataset_directory_bin}/{train_filename}.txt"
            train_path_num = f"{dataset_directory_num}/{train_filename}.txt"
            params["file"] = train_path_bin.replace(dataset_directory_bin, DATA_DIRECTORY)
            test_filename = params["test-file"]
            test_path_bin = f"{dataset_directory_bin}/{test_filename}.txt"
            test_path_num = f"{dataset_directory_num}/{test_filename}.txt"
            params["test-file"] = test_path_bin.replace(dataset_directory_bin, DATA_DIRECTORY)

            # Write STreeD files
            feature_names, num_extra_cols = make_streed_compatible(train_path_bin, train_path_num, params["file"])

            # Run STreeD
            time_duration, tree = run_streed(params, num_extra_cols)
            if time_duration >= 0:
                print(f"\033[33;1m{tree}\033[0m")
                print(f"\033[34mTime: \033[1m{time_duration:.3f}\033[0;34m seconds\033[0m")
            elif time_duration >= -1e8:
                print(f"\033[31mOut of time: \033[1m{-time_duration:.3f}\033[0;31m seconds\033[0m")
            else:
                print(
                    f"\033[30mIgnored due to Pareto front: {params['file'].split('/')[-1]} (depth = {params['max-depth']})\033[0m")

            # Parse tree string to lambda-structure
            feature_meanings = get_feature_meanings(params["core-file"])
            tree = serialize_tree_with_features(eval(tree), feature_names, feature_meanings)
            results.append((params, time_duration, tree))

            # Reset parameters to write to file nicely
            params["file"] = train_filename
            params["test-file"] = test_filename
    except KeyboardInterrupt:
        print("\033[33;1mHalted program!\033[0m")

    # Write trees to file
    f = open(f"{DIRECTORY}/output/coxstreedloglike_trees.csv", "w")
    f.write("id;settings;time;tree\n")
    for i, data in enumerate(results):
        f.write(f"{i};" + ";".join(str(j) for j in data) + "\n")
    f.close()

    total_end_time = time.time()
    print(f"\033[34mTotal time: \033[1m{total_end_time - total_start_time:.4f}\033[0;34m seconds")

    print("\033[32;1mDone!\033[0m")


if __name__ == "__main__":
    main()