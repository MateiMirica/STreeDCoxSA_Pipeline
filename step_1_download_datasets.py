import os
from SurvSet.data import SurvLoader
from utils import files_in_directory
from utils import ORIGINAL_DIRECTORY

def main():
    # Create necessary directories
    output_parent_directory = "/".join(ORIGINAL_DIRECTORY.split("/")[:-1])
    if not os.path.exists(output_parent_directory):
        os.mkdir(output_parent_directory)
    if not os.path.exists(ORIGINAL_DIRECTORY):
        os.mkdir(ORIGINAL_DIRECTORY)

    # Remove all previously generated datasets
    for filename in files_in_directory(ORIGINAL_DIRECTORY):
        if not filename.startswith("generated_dataset_"):
            os.remove(f"{ORIGINAL_DIRECTORY}/{filename}")

    loader = SurvLoader()
    for dataset_name in sorted(loader.df_ds["ds"]):
        df, _ = loader.load_dataset(ds_name=dataset_name).values()

        # Limit the amount of features in the dataset
        if df.shape[1] > 100:
            continue

        # Drop instances with NaN-values
        total_instances = df.shape[0]
        df = df.dropna()
        removed_instances = total_instances - df.shape[0]

        # Drop datasets with less than 2000 instances
        if df.shape[0] < 2000:
            continue

        # Change years into different format (this caused errors for OST)
        if dataset_name == "whas500":
            column = df["fac_year"]
            for i in range(len(column)):
                column[i] = column[i].split("-")[0]

        # Drop time2
        if "time2" in df.columns:
            df["time"] = df["time2"] - df["time"] # If time2 exists, time1 is the start time, so subtract and drop time2
            df.drop(columns=["time2"], inplace=True)

        # Write headers
        f = open(f"{ORIGINAL_DIRECTORY}/{dataset_name}.txt", "w")
        f.write("time,event,")
        feature_names = [j.replace(",", ".") for j in df][3:]
        f.write(",".join(feature_names).replace(":", "."))
        f.write("\n")

        # Write rows
        rows = [j[1] for j in df.iterrows()]
        for row in rows:
            time = max(1e-6, row["time"])
            event = int(row["event"] > 0.5)
            f.write(f"{time},{event},")
            f.write(",".join([str(row[key]).replace(",", ".") for key in feature_names]))
            f.write("\n")
        f.close()

        print(f"\033[35mDownloaded \033[1m{dataset_name}\033[0;35m (dropped \033[1m{removed_instances} / {total_instances}\033[0;35m instances)\033[0m")

if __name__ == "__main__":
    main()