import pandas as pd
import numpy as np
import peakutils
import logging
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, argrelextrema, find_peaks, peak_prominences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
M_PI = 3.141592


def get_unique_list_of_files(full_coordinates_path, ext_len):
    """
    Function to read all the filenames in a list
    """
    try:
        coordinates_files_list = os.listdir(full_coordinates_path)
        coordinates_files_list = [f[:-ext_len] for f in coordinates_files_list if not f.startswith(".")]
        unique_coordinates_files_list = list(set(coordinates_files_list))
    except Exception as e:
        logger.info("Error in getting the list of the files from: {} {}".format(full_coordinates_path, str(e)))
        unique_coordinates_files_list = None
    return unique_coordinates_files_list


def smooth_coordinates_sf(raw_coordinates):
    """
    Smooth the coordinates using the savgol filter
    """
    try:
        smooth_coordinates = savgol_filter(raw_coordinates, 31, 3)
        return smooth_coordinates
    except Exception as e:
        logger.error("Error in smoothening the signal: {}".format(str(e)))
        return None


def remove_baseline(coordinates):
    """
    Function to remove the baseline
    """
    try:
        baseline = peakutils.baseline(coordinates, 5)
        scaled = coordinates - baseline
        return scaled
    except Exception as e:
        logger.error("Error removing the baseline: {}".format(str(e)))
        return None


def plot_body_parts(df, pid, exercise_type, body_part, final_peaks):
    smooth_coordinates = smooth_coordinates_sf(df[body_part])
    index = np.arange(len(smooth_coordinates))
    plt.figure(figsize=(20, 5))
    plt.plot(smooth_coordinates)
    plt.plot(index[final_peaks], smooth_coordinates[final_peaks], 'ro', label='minima peaks')
    plt.title("{}_{}_{}".format(pid, exercise_type, len(final_peaks)))
    plt.savefig("/tmp/peaks3/{}_{}.jpg".format(pid, exercise_type))
    plt.close()


def plot_body_parts_all(df, pid, exercise_type, sensor_names, signal_names, final_peaks, sensor_body_parts_mapping):
    #     RW - > B8E3
    #     LW -> S57D0
    #     RA -> B8C7
    #     LA -> S548D
    column_list = df.columns.tolist()
    fig = plt.figure(figsize=(20, 10))
    count = 0
    for sensor_name in sensor_names:
        ax = fig.add_subplot(2, 2, count + 1)
        for signal_name in signal_names:
            for c in column_list:
                if sensor_name in c and signal_name in c and c.endswith("_Y"):
                    selected_col = c
            # logger.info("Plotting for ".format(len(filtered_peaks_max)))
            smooth_coordinates = np.array(smooth_coordinates_sf(df[selected_col]))
            raw_coordinates = np.array(df[selected_col])
            index = np.arange(len(raw_coordinates))
            _ = ax.plot(smooth_coordinates, label='smooth signal')
            # _ = ax.plot(raw_coordinates, label='raw signal')
            _ = ax.plot(index[final_peaks], smooth_coordinates[final_peaks], 'ro', label='minima peaks')
            _ = ax.set_title("{} {}".format(sensor_body_parts_mapping[sensor_name], signal_name))
            ax.legend()

        count += 1
    fig.suptitle("{}_{}_{}".format(pid, exercise_type, len(final_peaks)), fontsize=14)
    plt.tight_layout()
    plt.savefig("/tmp/peaks3/{}_{}.jpg".format(pid, exercise_type))
    plt.close()


def plot_body_parts_single_sensor(df, pid, exercise_type, sensor_name, final_peaks, sensor_body_parts_mapping):
    column_list = df.columns.tolist()
    fig = plt.figure(figsize=(20, 10))
    count = 0
    filter_col_names = []

    for c in column_list:
        if sensor_name in c:
            filter_col_names.append(c)

    for selected_col in filter_col_names:
        ax = fig.add_subplot(3, 3, count + 1)
        # logger.info("Plotting for ".format(len(filtered_peaks_max)))
        smooth_coordinates = np.array(smooth_coordinates_sf(df[selected_col]))
        raw_coordinates = np.array(df[selected_col])
        index = np.arange(len(raw_coordinates))
        _ = ax.plot(smooth_coordinates, label='smooth signal')
        # _ = ax.plot(raw_coordinates, label='raw signal')
        _ = ax.plot(index[final_peaks], smooth_coordinates[final_peaks], 'ro', label='minima peaks')
        _ = ax.set_title("{} {}".format(sensor_body_parts_mapping[sensor_name], selected_col))
        ax.legend()
        count += 1
    fig.suptitle("{}_{}_{}".format(pid, exercise_type, len(final_peaks)), fontsize=14)
    plt.tight_layout()
    plt.savefig("/tmp/peaks3/{}_{}.jpg".format(pid, exercise_type))
    plt.close()


def drop_columns(df, important_parts, drop_columns=None):
    try:
        column_list = df.columns.tolist()
        drop_columns_list = list(set(column_list) - set(important_parts))
        if drop_columns:
            drop_columns_list = drop_columns + drop_columns_list
        df = df.drop(drop_columns_list, axis=1)
        return df
    except Exception as e:
        logger.error("Error in dropping the columns: {}".format(str(e)))


def replace_with_mean(df):
    try:
        df = df.replace(0, df.mean())
        df = df.replace(np.nan, df.mean())
        return df
    except Exception as e:
        logger.error("Error in replacing with mean: {}".format(str(e)))


def merge(df_x, df_y):
    try:
        df_total = pd.DataFrame([], columns=df_x.columns)
        for each_column in df_x.columns:
            new_ = np.sqrt(df_x[each_column] * df_x[each_column] + df_y[each_column] * df_y[each_column])
            df_total[each_column] = new_
        return df_total
    except Exception as e:
        logger.error("Error in merging the x and y dataframe: {}".format(str(e)))
        return None


def merge_centre(df_x, df_y):
    try:
        df_total = pd.DataFrame([], columns=df_x.columns)
        mean_x = df_x[['LHip', 'RHip']].mean(axis=1)
        mean_y = df_y[['LHip', 'RHip']].mean(axis=1)

        for each_column in df_x.columns:
            new_ = np.sqrt((df_x[each_column] - mean_x) * (df_x[each_column] - mean_x) +
                           (df_y[each_column] - mean_y) * (df_y[each_column] - mean_y))
            df_total[each_column] = new_
        return df_total
    except Exception as e:
        logger.error("Error in merging (centric) the x and y dataframe: {}".format(str(e)))
        return None


def merge_helper(x_, y_, ignore_x, merge_type):
    df_merged = pd.concat([x_, y_], axis=1)
    return df_merged


def standardize_helper(df_merged, scaling_algo, important_parts):
    column_list = df_merged.columns.tolist()
    if scaling_algo == "znorm":
        for col in column_list:
            if col not in important_parts:
                continue
            df_merged[col] = (df_merged[col] - df_merged[col].mean()) / df_merged[col].std(ddof=0)

    if scaling_algo == "minmax":
        for col in column_list:
            if col not in important_parts:
                continue
            df_merged[col] = (df_merged[col] - df_merged[col].min()) / (df_merged[col].max() - df_merged[col].min())
    return df_merged


def standardize_np_array(combined_data):
    """
    Function to normalize a numpy matrix of matrix, for multivariate time series
    """
    number_records = combined_data.shape[0]
    scaled_combined_data = []
    for i in range(number_records):
        single_record = combined_data[i]
        single_record = (single_record - single_record.mean(axis=0)) / single_record.std(axis=0)
        scaled_combined_data.append(single_record)
    scaled_combined_data = np.array(scaled_combined_data)
    return scaled_combined_data


def create_custom_exercise_features(df):
    df["wrist_lr_shoulder_distance"] = np.sqrt(
        ((df["LWrist"] + df["LShoulder"]) / 2.0 - (df["RWrist"] + df["RShoulder"]) / 2.0) ** 2)
    df["hip_shoulder_distance"] = np.abs((df["LShoulder"] + df["RShoulder"]) / 2.0 - (df["LHip"] + df["RHip"]) / 2.0)
    df["wrist_shoulder_distance"] = np.min(
        (df["LWrist"] + df["RWrist"]) / 2.0 - (df["LShoulder"] + df["RShoulder"]) / 2.0)
    df["wrist_hip_distance"] = np.sqrt(((df["LWrist"] + df["RWrist"]) / 2.0 - (df["LHip"] + df["RHip"]) / 2.0) ** 2)
    return df


def custom_segment(df_merged, peaks_max_y, increment=1):
    sample_id = [-1] * df_merged.shape[0]
    count = 1
    # Use it to equally divide by total repetitions
    peaks_max_y[-1] -= 1
    for i in range(0, len(peaks_max_y) - 1, increment):
        starting = peaks_max_y[i]
        ending = -1
        if i + increment <= len(peaks_max_y) - 1:
            ending = peaks_max_y[i + increment]

        if ending > starting:
            for j in range(starting, ending):
                sample_id[j] = count
        count += 1
    return sample_id


def calculate_magnitude_info(shimmer_df, sensor_body_parts_mapping, sensor_type_signals):
    column_list = shimmer_df.columns.tolist()
    for sensor_name in sensor_body_parts_mapping:
        for sensor_type_signal in sensor_type_signals:
            selected_columns = []
            for c in column_list:
                if sensor_name in c and sensor_type_signal in c:
                    selected_columns.append(c)
            assert len(selected_columns) == 3
            shimmer_df["{}{}{}".format(sensor_name, sensor_type_signal, "Magnitude")] = np.sqrt(
                np.square(shimmer_df[selected_columns[0]]) + np.square(shimmer_df[selected_columns[1]]) + np.square(
                    shimmer_df[selected_columns[2]]))

    return shimmer_df


def apply_butter_worth_filter(ss, fc=20, sampling_freq=51.2, order=8):
    w = fc / (sampling_freq / 2)
    b, a = signal.butter(order, w, 'low')
    output = signal.filtfilt(b, a, ss)
    return output


def calculate_extra_features(shimmer_df, sensor_body_parts_mapping, sensor_type_signals):
    # accelerationX = (signed int)(((signed int)rawData_X) * 3.9)
    # accelerationY = (signed int)(((signed int)rawData_Y) * 3.9)
    # accelerationZ = (signed int)(((signed int)rawData_Z) * 3.9)

    column_list = shimmer_df.columns.tolist()
    for sensor_name in sensor_body_parts_mapping:
        selected_columns = []
        for c in column_list:
            if sensor_name in c and "_Accel_" in c and c.endswith(("X", "Y", "Z")):
                selected_columns.append(c)
        z_signal = y_signal = x_signal = ""
        for i in selected_columns:
            if i.endswith("_Z"):
                z_signal = i
            elif i.endswith("_Y"):
                y_signal = i
            else:
                x_signal = i

        shimmer_df["{}_{}".format(sensor_name, "pitch")] = 180 * np.arctan(
            shimmer_df[x_signal] / np.sqrt(
                shimmer_df[y_signal] * shimmer_df[y_signal] + shimmer_df[z_signal] * shimmer_df[z_signal])) / M_PI
        shimmer_df["{}_{}".format(sensor_name, "roll")] = 180 * np.arctan(
            shimmer_df[y_signal] / np.sqrt(
                shimmer_df[x_signal] * shimmer_df[x_signal] + shimmer_df[z_signal] * shimmer_df[z_signal])) / M_PI
        shimmer_df["{}_{}".format(sensor_name, "yaw")] = 180 * np.arctan(
            shimmer_df[z_signal] / np.sqrt(
                shimmer_df[x_signal] * shimmer_df[x_signal] + shimmer_df[z_signal] * shimmer_df[z_signal])) / M_PI
    return shimmer_df
