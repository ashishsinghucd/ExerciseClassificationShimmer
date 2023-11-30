import argparse
import configparser
import os
import logging
from pathlib import Path
import time
import pickle

from configobj import ConfigObj
from sklearn import metrics
import numpy as np
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import sktime
import pandas as pd

from utils.program_stats import timeit
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_NAME_X = '{}_{}_X'
FILE_NAME_Y = '{}_{}_Y'
FILE_NAME_PID = '{}_{}_pid'


def read_dataset(path):
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("TRAIN", data_type) + ".ts"))

    logger.info("Training data shape {} {} {}".format(x_train.shape, len(x_train.iloc[0, 0]), y_train.shape))
    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                FILE_NAME_X.format("TEST", data_type) + ".ts"))

    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))
    test_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TEST", data_type) + ".npy"), allow_pickle=True)
    train_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TRAIN", data_type) + ".npy"), allow_pickle=True)

    try:
        x_val, y_val = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("VAL", data_type) + ".ts"))
        logger.info("Validation data shape: {} {}".format(x_val.shape, y_val.shape))
    except (sktime.utils.data_io.TsFileParseException, FileNotFoundError):
        logger.info("Validation data is empty:")
        x_val, y_val = None, None

    return x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid


def combine_shimmer_hpe(path):
    logger.info("Loading the shimmer data from path: {}".format(path))
    x_train_shm, y_train_shm, x_test_shm, y_test_shm, x_val_shm, y_val_shm, train_pid_shm, test_pid_shm = read_dataset(
        path)

    hpe_path = path.replace('Shimmer', "HPE3")
    logger.info("Loading the HPE data from path: {}".format(hpe_path))
    x_train_hpe, y_train_hpe, x_test_hpe, y_test_hpe, x_val_hpe, y_val_hpe, train_pid_hpe, test_pid_hpe = read_dataset(
        hpe_path)
    train_shm_indices = train_pid_shm[:, 0].argsort()
    test_shm_indices = test_pid_shm[:, 0].argsort()

    train_hpe_indices = train_pid_hpe[:, 0].argsort()
    test_hpe_indices = test_pid_hpe[:, 0].argsort()

    x_train_shm = x_train_shm.iloc[train_shm_indices, :]
    y_train_shm = y_train_shm[train_shm_indices]

    x_test_shm = x_test_shm.iloc[test_shm_indices, :]
    y_test_shm = y_test_shm[test_shm_indices]

    x_train_hpe = x_train_hpe.iloc[train_hpe_indices, :]
    y_train_hpe = y_train_hpe[train_hpe_indices]

    x_test_hpe = x_test_hpe.iloc[test_hpe_indices, :]
    y_test_hpe = y_test_hpe[test_hpe_indices]

    x_train_shm = x_train_shm.reset_index(drop=True)
    x_test_shm = x_test_shm.reset_index(drop=True)

    x_train_hpe = x_train_hpe.reset_index(drop=True)
    x_test_hpe = x_test_hpe.reset_index(drop=True)

    x_train_shm_hpe = pd.concat([x_train_hpe, x_train_shm], axis=1)
    x_test_shm_hpe = pd.concat([x_test_hpe, x_test_shm], axis=1)

    train_pid_hpe = train_pid_hpe[train_hpe_indices]
    test_pid_hpe = test_pid_hpe[test_hpe_indices]

    logger.info("Training data shape {} {} {}".format(x_train_shm_hpe.shape, len(x_train_shm_hpe.iloc[0, 0]),
                                                      y_train_hpe.shape))

    logger.info("Testing data shape: {} {}".format(x_test_shm_hpe.shape, y_test_hpe.shape))

    return x_train_shm_hpe, y_train_hpe, x_test_shm_hpe, y_test_hpe, None, None, train_pid_hpe, test_pid_hpe


class RocketTransformerClassifier:
    def __init__(self, exercise):
        self.exercise = exercise
        self.classifiers_mapping = {}

    @timeit
    def fit_rocket(self, x_train, y_train, train_pid, kernels=10000):
        rocket = Rocket(num_kernels=kernels, normalise=False)  # random_state=100343
        rocket.fit(x_train)
        x_training_transform = rocket.transform(x_train)
        self.classifiers_mapping["transformer"] = rocket
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(x_training_transform, y_train)

        # Training Predictions
        # predictions = classifier.predict(x_training_transform)
        # d = classifier.decision_function(x_training_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)
        self.classifiers_mapping["classifier"] = classifier

        # self.create_prob_df(train_pid, y_train, predictions, probs, training_data=True)

    @timeit
    def predict_rocket(self, x_test, y_test, test_pid, x_val=None, y_val=None):
        rocket = self.classifiers_mapping["transformer"]
        classifier = self.classifiers_mapping["classifier"]
        x_test_transform = rocket.transform(x_test)

        # Test Predictions
        predictions = classifier.predict(x_test_transform)
        # d = self.classifiers_mapping["classifier"].decision_function(x_test_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)
        # self.create_prob_df(test_pid, y_test, predictions, probs)

        # Confusion Matrix
        labels = list(np.sort(np.unique(y_test)))
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing data")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
        logger.info("\n Classification report: \n{}".format(classification_report))

        classification_report_list.append(classification_report)

        plot_confusion_matrix(output_results_path, seed_value, confusion_matrix, labels)

        if x_val:
            logger.info("-----------------------------------------------")
            logger.info("Metrics on validation data")
            x_val_transform = rocket.transform(x_val)
            predictions = classifier.predict(x_val_transform)
            confusion_matrix = metrics.confusion_matrix(y_val, predictions)
            classification_report = metrics.classification_report(y_val, predictions)
            logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
            logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
            logger.info("\n Classification report: \n{}".format(classification_report))

    def create_prob_df(self, data_pid, y_test, predictions, probs, training_data=False):
        predictions_df = pd.DataFrame(probs, columns=self.classifiers_mapping["classifier"].classes_)
        predictions_df["PredictedLabel"] = predictions
        predictions_df["ActualLabel"] = y_test

        pid_info_df = pd.DataFrame(data_pid, columns=["Filename", "StartTime", "EndTime", "rep"])
        pid_info_df["PID"] = pid_info_df.apply(lambda x: x["Filename"].split(" ")[0], axis=1)
        pid_info_df["CorrectPrediction"] = y_test == predictions
        final_df = pd.concat([pid_info_df, predictions_df], axis=1)
        final_df = final_df.drop(["StartTime", "EndTime", "Filename"], axis=1)
        final_df["rep"] = final_df["rep"].astype(np.int64)
        f = "testing"
        if training_data:
            f = "training"
        # merge_df = merge_prob_confidence(final_df)
        # merge_df.to_csv('{}/probs_{}_{}.csv'.format(output_results_path, seed_value, f), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocket_config", required=True, help="path of the config file")
    args = parser.parse_args()
    rocket_config = ConfigObj(args.rocket_config)

    home_path = str(Path.home())
    seed_values = rocket_config["SEED_VALUES"]
    exercise = rocket_config["EXERCISE"]
    output_path = os.path.join(home_path, rocket_config["OUTPUT_PATH"])
    data_type = rocket_config["DATA_TYPE"]
    valid_classes = ["A", "Arch", "N", "R"]
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    base_path = os.path.join(home_path, rocket_config["BASE_PATH"], exercise)
    input_data_path = os.path.join(base_path, rocket_config["INPUT_DATA_PATH"])

    output_results_path = os.path.join(output_path, "Rocket")
    create_directory_if_not_exists(output_results_path)

    classification_report_list = []
    for seed_value in seed_values:
        logger.info("----------------------------------------------------")
        logger.info("Fitting Rocket for seed value: {}".format(seed_value))
        input_path_combined = os.path.join(input_data_path, seed_value, "MulticlassSplit")
        if not os.path.exists(input_path_combined):
            logger.info("Path does not exist for seed: {}".format(seed_value))
            continue
        x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid = combine_shimmer_hpe(input_path_combined)
        ts = time.time()
        rocket_classifier = RocketTransformerClassifier(exercise)
        rocket_classifier.fit_rocket(x_train, y_train, train_pid)
        rocket_classifier.predict_rocket(x_test, y_test, test_pid, x_val, y_val)
        te = time.time()
        total_time = (te - ts)
        logger.info('Total time preprocessing: {} seconds'.format(total_time))

    logger.info("Average classification report")
    logger.info(report_average(*classification_report_list))


