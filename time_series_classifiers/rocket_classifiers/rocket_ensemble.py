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


def read_dataset(path, data_type="default"):
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("TRAIN", data_type) + ".ts"))

    logger.info("Training data shape {} {} {}".format(x_train.shape, len(x_train.iloc[0, 0]), y_train.shape))
    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                FILE_NAME_X.format("TEST", data_type) + ".ts"))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))

    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))
    test_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TEST", data_type) + ".npy"), allow_pickle=True)
    train_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TRAIN", data_type) + ".npy"), allow_pickle=True)

    try:
        x_val, y_val = load_from_tsfile_to_dataframe(os.path.join(input_data_path,
                                                                  FILE_NAME_X.format("VAL", data_type) + ".ts"))
        logger.info("Validation data shape: {} {}".format(x_val.shape, y_val.shape))
    except (sktime.utils.data_io.TsFileParseException, FileNotFoundError):
        logger.info("Validation data is empty:")
        x_val, y_val = None, None

    return x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid


class RocketTransformerClassifierEnsemble:
    def __init__(self, exercise):
        self.exercise = exercise
        self.classifiers_mapping = {}

    @timeit
    def fit_rocket(self, x_train_shm, y_train_shm, train_pid_shm, x_train_hpe, y_train_hpe, train_pid_hpe,
                   kernels=10000):
        rocket_shm = Rocket(num_kernels=kernels, normalise=False)  # random_state=100343
        rocket_shm.fit(x_train_shm)
        x_training_transform_shm = rocket_shm.transform(x_train_shm)
        classifier_shm = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier_shm.fit(x_training_transform_shm, y_train_shm)
        self.classifiers_mapping["transformer_shm"] = rocket_shm
        self.classifiers_mapping["classifier_shm"] = classifier_shm

        rocket_hpe = Rocket(num_kernels=kernels, normalise=False)  # random_state=100343
        rocket_hpe.fit(x_train_hpe)
        x_training_transform_hpe = rocket_hpe.transform(x_train_hpe)
        classifier_hpe = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier_hpe.fit(x_training_transform_hpe, y_train_hpe)
        self.classifiers_mapping["transformer_hpe"] = rocket_hpe
        self.classifiers_mapping["classifier_hpe"] = classifier_hpe

        # Training Predictions
        # predictions = classifier.predict(x_training_transform)
        # d = classifier.decision_function(x_training_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)

        # self.create_prob_df(train_pid, y_train, predictions, probs, training_data=True)

    @timeit
    def predict_rocket(self, x_test_shm, y_test_shm, test_pid_shm, x_test_hpe, y_test_hpe, test_pid_hpe, x_val=None,
                       y_val=None):
        test_shm_indices = test_pid_shm[:, 0].argsort()
        test_hpe_indices = test_pid_hpe[:, 0].argsort()

        x_test_shm = x_test_shm.iloc[test_shm_indices, :]
        y_test_shm = y_test_shm[test_shm_indices]

        x_test_hpe = x_test_hpe.iloc[test_hpe_indices, :]
        y_test_hpe = y_test_hpe[test_hpe_indices]

        x_test_shm = x_test_shm.reset_index(drop=True)
        x_test_hpe = x_test_hpe.reset_index(drop=True)

        test_pid_shm = test_pid_hpe[test_shm_indices]
        test_pid_hpe = test_pid_hpe[test_hpe_indices]

        # np.testing.assert_array_equal(y_test_hpe, y_test_shm)


        rocket_shm = self.classifiers_mapping["transformer_shm"]
        classifier_shm = self.classifiers_mapping["classifier_shm"]
        x_test_transform_shm = rocket_shm.transform(x_test_shm)

        predictions_shm = classifier_shm.predict(x_test_transform_shm)

        d_shm = classifier_shm.decision_function(x_test_transform_shm)
        probs_shm = np.exp(d_shm) / np.sum(np.exp(d_shm), axis=1).reshape(-1, 1)

        confusion_matrix_shm = metrics.confusion_matrix(y_test_shm, predictions_shm)
        classification_report_shm = metrics.classification_report(y_test_shm, predictions_shm)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing data for Shimmer")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test_shm, predictions_shm)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix_shm))
        logger.info("\n Classification report: \n{}".format(classification_report_shm))


        rocket_hpe = self.classifiers_mapping["transformer_hpe"]
        classifier_hpe = self.classifiers_mapping["classifier_hpe"]
        x_test_transform_hpe = rocket_hpe.transform(x_test_hpe)

        # Test Predictions
        predictions_hpe = classifier_hpe.predict(x_test_transform_hpe)

        d_hpe = classifier_hpe.decision_function(x_test_transform_hpe)
        probs_hpe = np.exp(d_hpe) / np.sum(np.exp(d_hpe), axis=1).reshape(-1, 1)

        confusion_matrix_hpe = metrics.confusion_matrix(y_test_hpe, predictions_hpe)
        classification_report_hpe = metrics.classification_report(y_test_hpe, predictions_hpe)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing data for HPE")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test_hpe, predictions_hpe)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix_hpe))
        logger.info("\n Classification report: \n{}".format(classification_report_hpe))

        probs_combined = (probs_shm + probs_hpe)/2

        predictions_max_indices = np.argmax(probs_combined, axis=1)
        predictions_combined = [self.classifiers_mapping["classifier_hpe"].classes_[i] for i in predictions_max_indices]
        # Confusion Matrix
        confusion_matrix = metrics.confusion_matrix(y_test_hpe, predictions_combined)
        classification_report = metrics.classification_report(y_test_hpe, predictions_combined)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing combined data")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test_hpe, predictions_combined)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
        logger.info("\n Classification report: \n{}".format(classification_report))

        classification_report_list.append(classification_report)
        # self.create_prob_df(test_pid_hpe, y_test_hpe, predictions_combined, probs_combined)
        # plot_confusion_matrix(output_results_path, seed_value, confusion_matrix, labels)

    def create_prob_df(self, data_pid, y_test, predictions, probs, training_data=False):
        predictions_df = pd.DataFrame(probs, columns=self.classifiers_mapping["classifier_hpe"].classes_)
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
        final_df.to_csv('{}/probs_{}_{}.csv'.format(output_results_path, seed_value, f), index=False)
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
        x_train_shm, y_train_shm, x_test_shm, y_test_shm, x_val_shm, y_val_shm, train_pid_shm, test_pid_shm = read_dataset(
            input_path_combined, data_type)
        hpe_path = input_path_combined.replace('Shimmer', "HPE3")
        logger.info("Loading the HPE data from path: {}".format(hpe_path))
        x_train_hpe, y_train_hpe, x_test_hpe, y_test_hpe, x_val_hpe, y_val_hpe, train_pid_hpe, test_pid_hpe = read_dataset(
            hpe_path, "default")
        ts = time.time()
        rocket_classifier = RocketTransformerClassifierEnsemble(exercise)
        rocket_classifier.fit_rocket(x_train_shm, y_train_shm, train_pid_shm, x_train_hpe, y_train_hpe, train_pid_hpe)
        rocket_classifier.predict_rocket(x_test_shm, y_test_shm, test_pid_shm, x_test_hpe, y_test_hpe, test_pid_hpe)
        te = time.time()
        total_time = (te - ts)
        logger.info('Total time preprocessing: {} seconds'.format(total_time))

    logger.info("Average classification report")
    logger.info(report_average(*classification_report_list))

"""
rocket normalize done the same thing as standardizing the data, so we don't have to do it again
set it false from now on.

rocket on seed 103007
Training data shape  (1181, 8) (1181,)
Testing data shape:  (570, 8) (570,)
Validation data shape:  (191, 8) (191,)
STANDARDIZATION=True
SCALING_TYPE=znorm
INTERPOLATION=True
rocket false, classifier true
kernel, testing_accuracy, validation_accuracy
10, 0.43, 0.31
50, 0.53, 0.51
100, 0.55, 0.50
500, 0.59, 0.59
1000, 0.58, 0.63
5000, 0.61, 0.63
10000, 0.60, 0.63
50000, 0.60, 0.66
100000, 0.61, 0.65
110000, 0.61, 0.64
140000, 0.61, 0.64
150000, 0.62, 0.64
200000, 0.62, 0.65
"""
