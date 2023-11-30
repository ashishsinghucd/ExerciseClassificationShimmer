import argparse
import json
import os
import sys
import configparser
import traceback
from pathlib import Path
import logging
import warnings
import time

warnings.filterwarnings("ignore")

import seaborn as sns
import pandas as pd
import numpy as np
import peakutils
import matplotlib.pyplot as plt
from configobj import ConfigObj

from data_processing.create_segments.preprocess_utils import get_unique_list_of_files, plot_body_parts_single_sensor, \
    calculate_magnitude_info, calculate_extra_features
from data_processing.create_segments.preprocess_utils import drop_columns, smooth_coordinates_sf, \
    replace_with_mean, merge_helper, custom_segment, standardize_helper, plot_body_parts, plot_body_parts_all
from utils.util_functions import create_directory_if_not_exists, delete_directory_if_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('darkgrid')
segment_stats = []
ignored_clips_list = []
valid_files_count = 0
frames_info = {}
count_zeros = 0
total_shape = 0
total_mean_df = pd.DataFrame()
segment_temp = {"pid": [], "count": []}

sensor_body_parts_mapping = {"B8E3": "Right Wrist", "S57D0": "Left Wrist", "B8C7": "Right Arm", "S548D": "Left Arm",
                             "S9276": "Back"}
sensor_type_signals = ["_Gyro_", "_Accel_", "_Mag_"]
exercise_types_mapping = {"MP": ["A", "Arch", "N", "R"], "Rowing": ["A", "Ext", "N", "R", "RB"]}


def generate_segmented_coordinates(coordinate_base_path, segment_path):
    """
    Function to get the indices of the peaks. It also drops the unnecessary columns, rename the columns and replace
    empty values with the mean value. This functions kind of operates in a similar behaviour as for the video case.
    """
    stats = {}
    pid, exercise_type = coordinate_base_path.split(" ")[0], coordinate_base_path.split(" ")[2]
    check_file_name = "{}_{}".format(pid, exercise_type)
    if common_pids and check_file_name not in common_pids:
        return
    global valid_files_count
    valid_files_count += 1
    df = pd.read_csv(os.path.join(full_coordinates_path, coordinate_base_path + ".csv"))

    # Calculate extra features
    if calculate_mag:
        df = calculate_magnitude_info(df, sensor_body_parts_mapping, ["_Gyro_", "_Accel_"])
        df = calculate_extra_features(df, sensor_body_parts_mapping, sensor_type_signals)
    logger.info("Shape after calculating extra information: {} {}".format(df.shape[0], df.shape[1]))
    df = df.fillna(0)
    column_list = df.columns.tolist()
    # Filter the columns
    filter_columns = []

    # This loop is used in case we want to select a specific sensor
    # for c in column_list:
        #     """
        #     RW - > B8E3
        #     LW -> S57D0
        #     RA -> B8C7
        #     LA -> S548D
        #     Back -> S9276
        # _Gyro_
        # _Accel_
        # _Mag_
        #     """
        #
        # if "B8E3" in c:  # or "S548D" in c:  # and "_Gyro_" in c
        #     filter_columns.append(c)
    # if c.endswith("_Mag"):
    #     filter_columns.append(c)
    # elif "S548D" or "B8C7" in c:
    #     filter_columns.append(c)
    # elif "S2976" in c:
    #     filter_columns.append(c)
    # logger.info("Filter columns: {}".format(filter_columns))
    if not filter_columns:
        filter_columns = column_list
    df = df[filter_columns]
    # Copy the dataframes

    global count_zeros, total_shape
    count_zeros += df[df == 0].count(axis=1).sum()
    total_shape += df.size

    # Get the peaks based on a particular body part, it may vary for different exercises
    peaks_max_y = segment_info[pid][exercise_type]
    peaks_max_y.sort()
    # for i in range(len(peaks_max_y)):
    #     peaks_max_y[i] = int(peaks_max_y[i]*(51.2/30)) + shift_index
    reps_duration_list = [x - peaks_max_y[i - 1] for i, x in enumerate(peaks_max_y) if i > 0]
    logger.info(reps_duration_list)

    # Plot the body parts
    # plot_body_parts(df, pid, exercise_type, "Shimmer_B8C7_Mag_Y", peaks_max_y)  # Shimmer_B8C7_Mag_Y
    sensor_names = ["B8C7"]
    signal_names = ["Mag"]
    # plot_body_parts_all(df, pid, exercise_type, sensor_names, signal_names, peaks_max_y, sensor_body_parts_mapping)
    # plot_body_parts_single_sensor(df, pid, exercise_type, "B8E3", peaks_max_y, sensor_body_parts_mapping)
    segment_temp["pid"].append(check_file_name)
    segment_temp["count"].append(len(peaks_max_y) - 1)

    # Save the frame information for each person id
    if pid not in frames_info:
        frames_info[pid] = {}
    frames_info[pid][exercise_type] = [int(i) for i in peaks_max_y]

    logger.info("Total peaks are: {}".format(len(peaks_max_y)))

    df["frame_number"] = np.arange(df.shape[0])
    df["frame_peaks"] = df["frame_number"].isin(peaks_max_y).astype(int)
    df["pid"] = pid

    # Get the stats
    stats["pid"] = pid
    stats["exercise"] = exercise
    stats["exercise_type"] = exercise_type
    stats["number_of_reps"] = len(peaks_max_y) - 1
    stats["indices_peaks"] = peaks_max_y
    stats["duration_reps"] = reps_duration_list
    stats["average_reps_duration"] = np.round(np.mean(reps_duration_list), 2)
    stats["max_reps_duration"] = np.max(reps_duration_list)
    stats["min_reps_duration"] = np.min(reps_duration_list)
    segment_stats.append(stats)

    # Create a sample id for each segment
    sample_id = custom_segment(df, peaks_max_y, 1)
    df["sample_id"] = sample_id

    # Remove the negative sample ids
    df = df[df["sample_id"] != -1]
    df.to_csv(segment_path + "/" + "{}_{}".format(pid, exercise_type) + ".csv", index=False)


def smooth_remove_bs(column_list, df):
    """
    Function to remove the baseline signal from a signal
    """
    for col in column_list:
        if df[col].nunique() == 1:
            continue
        smooth_coordinates = smooth_coordinates_sf(df[col])
        baseline = peakutils.baseline(smooth_coordinates, 5)
        scaled = smooth_coordinates - baseline
        df.loc[:, col] = scaled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_test_config", required=True, help="path of the config file")
    args = parser.parse_args()

    train_test_config = ConfigObj(args.train_test_config)

    # Read the arguments
    home_path = str(Path.home())
    exercise = train_test_config["EXERCISE"]
    # Read the paths and folders
    segment_stats_dir = train_test_config["SEGMENT_STATS_DIR"]
    segmented_coordinates_dir = train_test_config["SEGMENTED_COORDINATES_DIR"]
    segment_info_file = train_test_config["SEGMENT_INFO_FILE"]
    common_pids = train_test_config["COMMON_PIDS"]
    calculate_mag = train_test_config.as_bool("MAGNITUDE_INFO")

    base_path = os.path.join(home_path, train_test_config["BASE_PATH"], exercise)
    full_coordinates_path = os.path.join(base_path, train_test_config["FULL_COORDINATES_PATH"])

    with open(os.path.join(segment_info_file)) as f:
        segment_info = json.load(f)

    full_segmented_coordinates_path = os.path.join(base_path, segmented_coordinates_dir)
    segment_stats_path = os.path.join(base_path, segment_stats_dir)

    delete_directory_if_exists(full_segmented_coordinates_path)
    create_directory_if_not_exists(full_segmented_coordinates_path)
    create_directory_if_not_exists(segment_stats_path)

    valid_classes = exercise_types_mapping[exercise]

    unique_coordinates_files = get_unique_list_of_files(full_coordinates_path, 4)
    if not unique_coordinates_files:
        logger.info("No coordinates files found")
        sys.exit(0)

    logger.info("Total number of coordinates files: {}".format(len(unique_coordinates_files)))

    ts = time.time()
    for coordinate_base_path in unique_coordinates_files:
        logger.info("Running for {}".format(coordinate_base_path))
        try:
            if not coordinate_base_path.startswith('.'):
                generate_segmented_coordinates(coordinate_base_path, full_segmented_coordinates_path)
        except Exception as e:
            logger.info("Error in generating the coordinates for: {} {}".format(coordinate_base_path, str(e)))
            logger.info(traceback.format_exc())
    te = time.time()
    total_time = (te - ts)
    logger.info('Total time preprocessing: {} seconds'.format(total_time))

    segment_stats_df = pd.DataFrame(segment_stats)

    stats_col_order = ["pid", "exercise", "exercise_type", "number_of_reps", "average_reps_duration",
                       "max_reps_duration", "min_reps_duration", "duration_reps", "indices_peaks"]
    segment_stats_ordered_df = segment_stats_df[stats_col_order]
    reps_list = segment_stats_ordered_df["duration_reps"].tolist()
    duration_list = []
    for sublist in reps_list:
        for duration in sublist:
            duration_list.append(duration)

    segment_stats_ordered_df = segment_stats_ordered_df[segment_stats_ordered_df["exercise_type"].isin(valid_classes)]

    total_person = len(segment_stats_ordered_df["pid"].unique())
    total_exercise_type = len(segment_stats_ordered_df["exercise_type"].unique())
    total_samples = segment_stats_ordered_df["number_of_reps"].sum()
    overall_average = round(segment_stats_ordered_df["average_reps_duration"].mean(), 2)
    overall_max = segment_stats_ordered_df["max_reps_duration"].max()
    overall_min = segment_stats_ordered_df["min_reps_duration"].min()
    overall_stats = ["total_person: " + str(total_person),
                     None,
                     "total_exercise_type: " + str(total_exercise_type),
                     "total_samples: " + str(total_samples),
                     "Average duration of rep:" + str(overall_average),
                     "Maximum duration of rep:" + str(overall_max),
                     "Minimum duration of rep:" + str(overall_min),
                     None,
                     None]

    segment_stats_ordered_df = segment_stats_ordered_df.sort_values(by=['pid'])
    segment_stats_ordered_df.index.name = "row_number"
    segment_stats_ordered_df.loc['data_info'] = overall_stats
    file_name = "segment_stats"
    current_directory = os.path.join(os.getcwd(), 'stats')
    plt.figure(figsize=(10, 3))
    sns.distplot(duration_list)
    plt.title('Histogram for the duration of reps')
    plt.xlabel('Counts')
    plt.ylabel('Duration')
    plt.savefig(segment_stats_path + "/duration_hist.png")
    segment_stats_ordered_df[stats_col_order].to_csv(segment_stats_path + '/{}.csv'.format(file_name), index=True)
    logger.info("Total number of coordinates files: {}".format(len(unique_coordinates_files)))
    logger.info("Total valid files are: {}".format(valid_files_count))
    logger.info("Total ignored files are: {}".format(len(ignored_clips_list)))
    logger.info("Total proportion of zeros: {}".format(float(count_zeros) * 100 / total_shape))

    segment_temp_df = pd.DataFrame(segment_temp)
    segment_temp_df.to_csv("/tmp/segment_shimmer.csv", index=False)

