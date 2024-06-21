import Pkg; Pkg.add("CSV")
import Pkg; Pkg.add("DataFrames")
import Pkg; Pkg.add("JSON")
println("\033[30;1mStarted\033[0m")
total_start_time = time()

DIRECTORY = dirname(Base.source_path())
DATASET_TYPE = "original"
#DATASET_TYPE = "binary"

# Import modules (takes long)
using CSV
using DataFrames
using JSON
#IAISystemImages.install()
println("\033[30;1mImported modules\033[0m")

# Serialize the tree learner to a lambda-structure
#
# tree              The learner object to serialize
# node_idx          The index of the node to convert at the moment (root = 1)
# feature_meanings  The dictionary to read feature meanings from
function serialize_tree(tree, node_idx, feature_meanings)
    if IAI.is_leaf(tree, node_idx)
        # Leaf node
        return "[None]"
    else
        # Decision node
        feature_name = string(IAI.get_split_feature(tree, node_idx))

        feature_description = ""
        swap = false
        if haskey(feature_meanings, feature_name)
            # If the feature meaning is found in the map, use it
            feature_description = feature_meanings[feature_name]

            # Set swap to false, since OST gives thresholds like "... < 0.5", thereby negating the meaning of the feature
            swap = true
        else
            # If the feature meaning isn't found, create an own lambda
            attribute = "x['" * feature_name * "']"
            feature_description = "lambda x: " * attribute

            if IAI.is_parallel_split(tree, node_idx)
                # Use "... < ..." for numerical features
                feature_description *= " < " * string(IAI.get_split_threshold(tree, node_idx))
            else
                # Use "... in [...]" for categorical features
                categories = IAI.get_split_categories(tree, node_idx)
                included_categories = []
                for key in keys(categories)
                    if categories[key]
                        name = key
                        if tryparse(Float64, name) === nothing
                            name = "'" * replace(name, "'" => "\\'") * "'"
                        end
                        push!(included_categories, name)
                    end
                end

                feature_description *= " in [" * join(included_categories, ",") * "]"
            end
        end

        # Serialize children
        lower_child = serialize_tree(tree, IAI.get_upper_child(tree, node_idx), feature_meanings)
        upper_child = serialize_tree(tree, IAI.get_lower_child(tree, node_idx), feature_meanings)

        # Swap children if necessary
        if swap
            lower_child, upper_child = upper_child, lower_child
        end

        return "[" * feature_description *
            "," * lower_child *
            "," * upper_child * "]"
    end
end

results = []
open(DIRECTORY * "/output/settings.txt") do f
    # Default parameters for warming up
#     file = "generated_dataset_02000_1_80_0"
    file = "divorce"
    core_file = file
    max_depth = 0
    max_num_nodes = 0
    cost_complexity = 0
    hypertuning = false

    continuing = true
    warming_up = true
    settings_line = nothing

    while true
        continuing = !eof(f)

        # Read data
        df = CSV.read(DIRECTORY * "/datasets/" * DATASET_TYPE * "/" * file * ".txt", DataFrame, pool=true)
        events = df[!, "event"] .== 1
        times = df[!, "time"]
        X = select!(df, Not([:event, :time]))

        # Disable garbage collect during fitting process
        GC.gc()
        GC.enable(false)

        if warming_up
            println("\033[35;1mWarming up\033[0m")
        else
            println("\033[35;1mRunning " * file * ".txt {depth = " * string(max_depth) * "}\033[0m")
        end
        random_seed = 1
        object_to_fit = nothing
        start_time = end_time = 0
        already_halted_once = false
        while true
            try
                # Define learner object
                if hypertuning
                    object_to_fit = IAI.GridSearch(
                        IAI.OptimalTreeSurvivalLearner(
                            random_seed=random_seed,
                            skip_curve_fitting=false,
                            death_minbucket=0,
                        ),
                        max_depth=0:max_depth,
                        n_folds=10,
                    )
                else
                    object_to_fit = IAI.OptimalTreeSurvivalLearner(
                        random_seed=random_seed,
                        skip_curve_fitting=false,
                        cp=cost_complexity,
                        death_minbucket=0,
                        minbucket=1,
                        max_depth=max_depth,
                    )
                end

                # Start training
                start_time = time()
                IAI.fit!(object_to_fit, X, events, times)
                end_time = time()

                break
            catch ex
                # If something goes wrong, check whether it was intentional
                if isa(ex, InterruptException)
                    if already_halted_once
                        println("\033[33;1mHalted program!\033[0m")
                        break
                    end

                    println("\033[33mKeyboard interrupt! Do it again to halt the program.\033[0m")
                    already_halted_once = true
                    continue
                end

                # Try again with a different seed
                println(ex)
                println("\033[31;1mAn error occurred, retrying with a different seed\033[0m")
                random_seed += 1
            end

            already_halted_once = false
        end
        GC.enable(true)

        # Get tree learner and thetas
        learner = object_to_fit
        if hypertuning
            learner = IAI.get_learner(object_to_fit)
        end
        thetas = IAI.predict_hazard(learner, X)
        println(learner)

        if !warming_up
            # Read feature meanings
            feature_meanings = Dict()
            open(DIRECTORY * "/datasets/feature_meanings/" * core_file * ".txt") do fmf
                while !eof(fmf)
                    text = readline(fmf)
                    equals_idx = findfirst(" = ", text).start
                    key = text[1:(equals_idx - 1)]
                    value = text[(equals_idx + 3):end]
                    feature_meanings[key] = value
                end
            end

            # Parse tree and push results
            tree_string = serialize_tree(learner, 1, feature_meanings)
            time_duration = round(end_time - start_time, digits=3)
            push!(results, (settings_line, time_duration, tree_string))

            println("\033[34;1mTime: ", time_duration, " seconds\033[0m\n")
        else
            warming_up = false
        end

        if !continuing
            break
        end

        # Read the settings for the next run
        settings_line = readline(f)
        settings = JSON.parse(settings_line)

        file = get!(settings, "file", "404")
        core_file = get!(settings, "core-file", "404")
        max_depth = get!(settings, "max-depth", 3)
        max_num_nodes = get!(settings, "max-num-nodes", 7)
        cost_complexity = get!(settings, "cost-complexity", 0)
        hypertuning = get!(settings, "hyper-tune", "1") == "1"
    end
end

# Write trees to file
id = 0
open(DIRECTORY * "/output/ost_trees.csv", "w") do f
    write(f, "id;settings;time;tree\n")

    for data in results
        global id

        settings_line, time_duration, tree_string = data
        line = string(id) * ";" * settings_line * ";" * string(time_duration) * ";" * tree_string * "\n"
        write(f, line)

        id += 1
    end
end

total_end_time = time()
println("\n\033[34;1mTotal time: ", total_end_time - total_start_time, " seconds")
println("\033[32;1mDone!\033[0m")
