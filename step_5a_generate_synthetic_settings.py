import os
from step_4_split_datasets import K
from utils import files_in_directory
from utils import DIRECTORY, ORIGINAL_DIRECTORY


# Takes the cartesian product of a list of options per parameter
#
# items     A list of pairs containing a key and its list of possible values
def cartesian_product(items, idx=0):
    # Base case
    if idx == len(items):
        return [{}]

    # For each option in the current
    key, values = items[idx]
    results = []
    for value in values:
        for result in cartesian_product(items, idx + 1):
            result[key] = value
            results.append(result)
    return results


# Turns the given parameters to a list of parameter settings that can be passed to the algorithms
#
# parameters    A map containing the settings for the given experiment
def prepare_parameters(parameters):
    # Need to specify file
    if "file" not in parameters:
        raise Exception("Forgot to specify file")

    # Default test file to train file
    if "test-file" not in parameters:
        sp = parameters["file"].split("_")
        sp[2] = "06000"
        parameters["test-file"] = "_".join(sp)

    # Default max depth to 3
    if "max-depth" not in parameters:
        parameters["max-depth"] = 3

    # Default max num nodes to the maximum possible
    if "max-num-nodes" not in parameters:
        parameters["max-num-nodes"] = "max"
    if parameters["max-num-nodes"] == "max":
        parameters["max-num-nodes"] = 2 ** parameters["max-depth"] - 1

    # Default cost complexity to 0
    if "cost-complexity" not in parameters:
        parameters["cost-complexity"] = 0

    # Save core-filename
    parameters["core-file"] = parameters["file"]

    # If desired, use the k-fold splits of the given dataset instead
    if "split" not in parameters:
        parameters["split"] = False
    if parameters.pop("split"):
        parameters["file"] = "train/" + parameters["file"]
        parameters["test-file"] = "test/" + parameters["test-file"]

        result = []
        for idx in range(K):
            p = parameters.copy()
            p["file"] += f"_partition_{idx}"
            p["test-file"] += f"_partition_{idx}"
            result.append(p)

        return result

    return [parameters]


def main():
    PARAM_OPTIONS = {
        "file": [
            f"{j[:-4]}" for j in files_in_directory(ORIGINAL_DIRECTORY) if
            j.startswith("generated_dataset") and not j.startswith("generated_dataset_06000")
        ],
        "max-depth": [3],
        "max-num-nodes": ["max"],
        "cost-complexity": [0],
        "hyper-tune": ["1"],
        "split": [False],
    }

    # Create directory for settings files
    if not os.path.exists(f"{DIRECTORY}/output"):
        os.mkdir(f"{DIRECTORY}/output")

    # Turn given options into a list of parameters
    parameter_combinations = cartesian_product([*PARAM_OPTIONS.items()])
    prepared_parameter_combinations = []
    for parameters in parameter_combinations:
        parameters = prepare_parameters(parameters)
        prepared_parameter_combinations.extend(parameters)

    # Write to file
    f = open(f"{DIRECTORY}/output/settings.txt", "w")
    for parameters in prepared_parameter_combinations:
        f.write(f"{parameters}\n".replace("'", "\""))
        print(f"\033[35;1m{parameters}\033[0m")
    f.close()

    print("\033[32;1mDone!\033[0m")


if __name__ == "__main__":
    main()