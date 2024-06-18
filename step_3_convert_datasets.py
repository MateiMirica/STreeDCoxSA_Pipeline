import os
import shutil
from utils import files_in_directory, parse_line
from utils import DIRECTORY, ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY
from collections import Counter

MAX_BINARIZATIONS_PER_FEATURE = 10

# Checks for binary features that make the same split and removes one of them, and removes binary features for which all instances have the same value
# Returns the new feature names (with some features removed) and the new instances
#
# feature_names     The names of the features in order
# instances         The instances of the dataset
def remove_redundant_binary_features(feature_names, instances):
    # Create new instances with time and event already added
    new_feature_names = ["time", "event"]
    new_instances = [j[:2] for j in instances]

    # The binary value sequences already seen (000...000 is uninformative and therefore already included)
    seen = set([
        "0" * (len(instances[0]) - 1)
    ])

    for i in range(2, len(instances[0])):
        values = [inst[i] for inst in instances]

        # Check whether the variable is binary
        if len(set(values) - set([0, 1])) == 0:
            key = "".join(str(inst[i]) for inst in instances)
            complement_key = "".join(str(1 - inst[i]) for inst in instances)

            # Check whether this binary value sequence already exists in the dataset
            # If so, skip it
            if key in seen or complement_key in seen:
                continue
            seen.add(key)
            
            true_sum = sum([inst[i] for inst in instances])
            if true_sum <= 0.01 * len(instances) or true_sum >= 0.99 * len(instances):
                continue # Skip because this feature represents less than 1% of the data

        # Leave the variable in the dataset
        new_feature_names.append(feature_names[i])
        for k in range(len(new_instances)):
            new_instances[k].append(instances[k][i])
    num_old_features = len(instances[0]) - 2
    num_new_features = len(new_instances[0]) - 2
    if num_new_features < num_old_features:
        print(f"Removed {num_old_features - num_new_features} redundant or non-informative features")
    return new_feature_names, new_instances

# Turns the categorical features in a dataset to binary features
# Returns the new feature names, the new instances, and a mapping from new feature names to their meanings
#
# feature_names     The names of the features in order
# instances         The instances of the dataset
def turn_numeric(feature_names, instances):
    new_feature_names = ["time", "event"]
    new_instances = [j[:2] for j in instances]
    new_feature_meanings = {}

    converted_features_amount = 0
    for i in range(2, len(instances[0])):
        values = set([inst[i] for inst in instances])

        # Check whether this variable is not a numeric variable
        if any(type(j) == str for j in values):
            values = sorted(values, key=lambda x: str(x))

            # If there are only two options, comparing to one of them will be enough
            if len(values) == 2:
                values = [values[0]]
            elif len(values) > MAX_BINARIZATIONS_PER_FEATURE:
                counts = Counter([inst[i] for inst in instances])
                values = []
                last_value = []
                for j, (key, val) in enumerate(counts.most_common()):
                    if j < MAX_BINARIZATIONS_PER_FEATURE - 1:
                        values.append(key)
                    else:
                        last_value.append(key)
                values.append(last_value)
            
            def escape(obj):
                if isinstance(obj, list):
                    return [escape(j) for j in obj]
                elif isinstance(obj, str):
                    return obj.replace("'", "\\'")
                else:
                    return obj

            for option in values:
                option_escaped = escape(option)

                # Create binary variables using this option
                for k in range(len(instances)):
                    if isinstance(option, list):
                        new_instances[k].append(int(instances[k][i] in option))
                    else:
                        new_instances[k].append(int(instances[k][i] == option))

                # Save data
                new_feature_name = f"CatFeat{converted_features_amount}"
                new_feature_names.append(new_feature_name)
                if isinstance(option, list):
                    new_feature_meanings[new_feature_name] = f"lambda x: x['{feature_names[i]}'] in [" \
                        + ",".join([(f"'{o}'" if isinstance(o, str) else str(o)) for o in option_escaped]) + "]"
                elif isinstance(option, str):
                    new_feature_meanings[new_feature_name] = f"lambda x: x['{feature_names[i]}'] == '{option_escaped}'"
                else:
                    new_feature_meanings[new_feature_name] = f"lambda x: x['{feature_names[i]}'] == {option_escaped}"
                converted_features_amount += 1
        else:
            # Leave the variable for what it is
            new_feature_names.append(feature_names[i])
            for k in range(len(new_instances)):
                new_instances[k].append(instances[k][i])

    return new_feature_names, new_instances, new_feature_meanings

# Turns the continuous features in a dataset to binary features
# Returns the new feature names, the new instances, and a mapping from new feature names to their meanings
#
# feature_names     The names of the features in order
# instances         The instances of the dataset
def turn_binary(feature_names, instances):
    new_feature_names = ["time", "event"]
    new_instances = [j[:2] for j in instances]
    new_feature_meanings = {}

    converted_features_amount = 0
    for i in range(2, len(instances[0])):
        values = [inst[i] for inst in instances]

        # Check whether this variable is not a binary variable
        if set(values) - set([0, 1]):
            values = sorted([j for j in values])

            # Create a list of thresholds between values
            thresholds = [(values[j] + values[j + 1]) / 2 for j in range(len(values) - 1)]
            if len(set(thresholds)) > MAX_BINARIZATIONS_PER_FEATURE:
                # Reduce the amount of thresholds to a certain maximum
                indices = [round(((j + 0.5) * (len(thresholds) - 1)) / MAX_BINARIZATIONS_PER_FEATURE) for j in range(MAX_BINARIZATIONS_PER_FEATURE)]
                thresholds = [thresholds[j] for j in indices]
                
            thresholds = sorted(list(set(thresholds)))
            

            for threshold in thresholds:
                # Create binary variables using this threshold
                for k in range(len(instances)):
                    new_instances[k].append(int(instances[k][i] > threshold))

                # Save data
                new_feature_name = f"NumFeat{converted_features_amount}"
                new_feature_names.append(new_feature_name)
                new_feature_meanings[new_feature_name] = f"lambda x: x['{feature_names[i]}'] > {threshold}"
                converted_features_amount += 1
        else:
            # Leave the variable for what it is
            new_feature_names.append(feature_names[i])
            for k in range(len(new_instances)):
                new_instances[k].append(instances[k][i])

    return new_feature_names, new_instances, new_feature_meanings

def main():
    # Create a folder for binary feature meanings
    if os.path.exists(f"{DIRECTORY}/datasets/feature_meanings"):
        shutil.rmtree(f"{DIRECTORY}/datasets/feature_meanings")
    os.mkdir(f"{DIRECTORY}/datasets/feature_meanings")

    for output_directory in [NUMERIC_DIRECTORY, BINARY_DIRECTORY]:
        # Remove all previously converted datasets
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        for filename in files_in_directory(output_directory):
            os.remove(f"{output_directory}/{filename}")

    for input_filename in files_in_directory(ORIGINAL_DIRECTORY):
        name = input_filename[:-4]

        # Read instances from file
        f = open(f"{ORIGINAL_DIRECTORY}/{input_filename}")
        lines = f.read().strip().split("\n")
        f.close()

        instances = []
        feature_names = []
        feature_names = lines[0].split(",")
        for line in lines[1:]:
            inst = parse_line(line)
            instances.append(inst)

        # Convert to numeric instances
        numeric_feature_names, numeric_instances, numeric_feature_meanings = turn_numeric(feature_names, instances)
        numeric_feature_names, numeric_instances = remove_redundant_binary_features(numeric_feature_names, numeric_instances)

        # Write numeric instances
        f = open(f"{NUMERIC_DIRECTORY}/{name}.txt", "w")
        f.write(",".join(numeric_feature_names))
        f.write("\n")
        for num_inst in numeric_instances:
            f.write(",".join(str(j) for j in num_inst))
            f.write("\n")
        f.close()

        # Convert to binary instances
        binary_feature_names, binary_instances, binary_feature_meanings = turn_binary(numeric_feature_names, numeric_instances)
        binary_feature_names, binary_instances = remove_redundant_binary_features(binary_feature_names, binary_instances)
        for key, value in numeric_feature_meanings.items():
            binary_feature_meanings[key] = value

        # Write binary instances
        f = open(f"{BINARY_DIRECTORY}/{name}.txt", "w")
        f.write(",".join(binary_feature_names))
        f.write("\n")
        for bin_inst in binary_instances:
            f.write(",".join(str(j) for j in bin_inst))
            f.write("\n")
        f.close()

        # Write binary feature meanings
        f = open(f"{DIRECTORY}/datasets/feature_meanings/{name}.txt", "w")
        for key, value in binary_feature_meanings.items():
            f.write(f"{key} = {value}")
            f.write("\n")
        f.close()

        # Print progress
        print(f"\033[35mConverted \033[1m{str(input_filename)}\033[0;35m (\033[1m{len(binary_instances)}\033[0;35m instances)\033[0m")
        print(f"\033[34m  - Original    \033[1m{len(instances[0]) - 2}\033[0;34m features\033[0m")
        print(f"\033[34m  - Numeric     \033[1m{len(numeric_instances[0]) - 2}\033[0;34m features\033[0m")
        print(f"\033[34m  - Binary      \033[1m{len(binary_instances[0]) - 2}\033[0;34m features\033[0m")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
