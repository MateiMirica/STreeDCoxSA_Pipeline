library("hash")
library("jsonlite")
library("partykit")
library("readr")
library("survival")
library("mlr3")
library("mlr3proba")
library("mlr3extralearners")
library("mlr3tuning")

directory <- getwd()
dataset_type = "numeric"
#dataset_type = "binary"
TIME_OUT_IN_SECONDS <- 2000

settings_file <- paste(directory, "/output/settings.txt", sep = "")

total_start_time <- Sys.time()

# Serialize the tree learner to a lambda-structure
#
# tree              The learner object to serialize
# node_idx          The index of the node to convert at the moment (root = 1)
# feature_meanings  The dictionary to read feature meanings from
serialize_tree_with_features <- function(lines, root_idx, feature_meanings) {
  line <- lines[root_idx]
  depth <- lengths(regmatches(line, gregexpr("\\|   ", line)))

  # Leaf node
  if (root_idx == length(lines))
    return("[None]")

  # Find left and right children
  left_idx <- 0
  right_idx <- 0
  for (i in (root_idx + 1):length(lines)) {
    pattern_descendant <- strrep("|   ", depth + 1)
    if (!startsWith(lines[i], pattern_descendant))
      break

    pattern_child <- paste(pattern_descendant, "[", sep = "")
    if (startsWith(lines[i], pattern_child)) {
      if (left_idx == 0) {
        left_idx <- i
      } else {
        right_idx <- i
        break
      }
    }
  }

  # If `left_idx` was never updated, no child exists: leaf node
  if (left_idx == 0)
    return("[None]")

  # Read feature from the right child's condition
  feature_line <- lines[right_idx]

  # Find the splitting criterium
  start_idx <- regexpr("\\]", feature_line)[1] + 2
  end_idx <- nchar(feature_line)
  if (substring(feature_line, end_idx) == ")")
    end_idx <- regexpr("\\:[^\\:]*$", feature_line)[1] - 1
  criterium <- substring(feature_line, start_idx, end_idx)

  # Split criterium in feature name and comparison
  split_idx <- regexpr(" [^ ]* [^ ]*$", criterium)[1]
  feature_name <- substring(criterium, 1, split_idx - 1)
  comparison <- substring(criterium, split_idx)

  if (has.key(feature_name, feature_meanings)) {
    # If the feature meaning is already known, use it
    feature <- feature_meanings[[feature_name]]
  } else {
    # If not, create own lambda using the comparison
    feature <- paste("lambda x: x['", feature_name, "']", comparison, sep = "")
  }

  # Serialize children
  left_child <- serialize_tree_with_features(lines, left_idx, feature_meanings)
  right_child <- serialize_tree_with_features(lines, right_idx, feature_meanings)

  return(paste("[", feature, ",", left_child, ",", right_child, "]", sep = ""))
}

run_ctree <- function(settings) {
  name <- settings[["file"]]
  core_name <- settings[["core-file"]]
  max_depth <- settings[["max-depth"]]
  hypertuning <- settings[["hyper-tune"]] == "1"

  train_filename <- paste(dataset_type, settings["file"], sep = "/")

  # Read train data
  file_path <- paste(directory, "/datasets/", train_filename, ".txt", sep = "")
  sdata <- read.csv(file_path, stringsAsFactors = FALSE)
  # Fit and capture tree
  set.seed(1)

  duration = NULL
  tree = NULL
  if (hypertuning) {
    surv.task = TaskSurv$new(id=train_filename, backend=sdata, time='time', event='event')
    surv.task$col_roles$stratum = "event"

    surv.lrn = lrn("surv.ctree",
      maxdepth  = max_depth,
      mincriterion = to_tune(c(0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999))
    )

    instance = ti(
      task = surv.task,
      learner = surv.lrn,
      resampling = rsmp("cv", folds = 10),
      measures = msr("surv.cindex"),
      terminator = trm("run_time", secs=TIME_OUT_IN_SECONDS)
    )
    tuner = tnr("grid_search", resolution = 8)

    start_time <- Sys.time()
    print("Tune model")
    tuner$optimize(instance)

    print("Train tuned model")
    surv.lrn$param_set$values = instance$result_learner_param_vals
    surv.lrn$train(surv.task)
    tree <- surv.lrn$model

    end_time <- Sys.time()
    duration = difftime(end_time, start_time, units="secs")
    print(paste("Time:", duration))
  } else {
    start_time <- Sys.time()
    tree <- ctree(
      Surv(time, event) ~ .,
      data = sdata,
      control = ctree_control(
        maxdepth = max_depth
      )
    )
    end_time <- Sys.time()
    duration = difftime(end_time, start_time, units="secs")
  }

  tree_lines <- capture.output(print(tree))

  # Parse tree
  idx <- match(c("Fitted party:"), tree_lines)
  tree_lines <- tree_lines[(idx + 1):(length(tree_lines) - 3)]

  # Read feature meanings
  feature_meanings_file_path <- paste(directory, "/datasets/feature_meanings/", core_name, ".txt", sep = "")
  feature_meanings_lines <- read_lines(feature_meanings_file_path)

  # Parse feature meanings
  feature_meanings <- hash()
  for (feature_meanings_line in feature_meanings_lines) {
    split_idx <- regexpr(" = ", feature_meanings_line)[1]
    key <- substring(feature_meanings_line, 1, split_idx - 1)
    value <- substring(feature_meanings_line, split_idx + 3)
    feature_meanings[[key]] <- value
  }
  tree_string <- serialize_tree_with_features(tree_lines, 1, feature_meanings)

  print(tree)
  print(train_filename)

  return(list(duration, tree_string))
}

id <- 0
output_lines <- "id;settings;time;results"

settings_lines <- read_lines(settings_file)
for (settings_line in settings_lines) {
  # Read settings
  settings <- fromJSON(settings_line)
  result <- run_ctree(settings)
  duration <- result[[1]]
  tree_string <- result[[2]]

  # Append results
  output_line <- paste("\n",
    paste(id, settings_line, duration, tree_string, sep = ";"),
    sep = ""
  )
  output_lines <- paste(output_lines, output_line)

  id <- id + 1
}

# Write trees to file
sink(paste(directory, "/output/ctree_trees.csv", sep = ""))
cat(output_lines)
sink()

total_end_time <- Sys.time()
print(paste("Total time:", total_end_time - total_start_time))

print("Done!")