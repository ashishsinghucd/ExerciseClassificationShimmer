### Exploration and Analysis of Shimmer IMU data for Human Exercise Classification for MP and Rowing exercises.

This repository contains the code to generate train/test data, run the classification using the ROCKET or deep learning
classifiers using data from IMU Sensors.

Folders and their functionalities

1. `data_processing/create_segments`: used to create the segmentation using the peaks information. The main file is
`preprocess_coordinates_new.py` used to store the peak information in the dataframe.
2. `data_processing/create_train_test_data`: used to create the final train/test/val split for classification. 
3. `time_series_classifiers/deep_learning_classifiers`: used to run the deep learning models such as fcn or resnet to 
classify the data.
4. `time_series_classifiers/time_series_classifiers`: used to run the ROCKET based models to classify the data.
5. `utils`: contains basic utilities functions. 
6. `data_info`: contains files which have demographic info, peaks info for MP and Rowing exercises.


## Sensor Locations for MP
    - 548D – L Arm/Shank
    - B8C7 – R Arm/Shank
    - 9276 – Lspx
    - B8E3 – R Wrist
    - 57D0 – L Wrist

- The sampling rate for the data is usually 51.2 Hz in this data set but it is different for some exercises e.g. jump data. You will always be able to check this by checking a timestamp column in the files. They are in ms so you can look at 2 sequential samples and get sampling rate by looking at 2 sequential samples and using the formula:
1000/(sample(2)-sample(1))
- Reading the data is quite easy. We just have to rename the file to .csv extension.
- Description 5 sensors (lower back, 2 wrists, 2 elbows)
- Accelerometer (movement acceleration)
  - Calibrated and Uncalibrated 
    - We have to use the calibrated one 
  - Low noise and wide range 
    - These two are approximately the same. We only have to use the LN. 
  - X , Y and Z coordinates
- Magnetometer (orientation/compass?)
  - Calibrated and Uncalibrated 
  - X , Y and Z coordinates
- Gyroscope (angular velocity, degrees per second)
  - Calibrated and Uncalibrated 
  - X , Y and Z coordinates
- Magnetometer can be used to do the segmentation
- Each sensor 33 columns 
  - 12 from accelerometer 
  - 6 from magnetometer 
  - 6 from gyroscope 
  - Other columns 
    - Batt_percentage - 1 
    - Event Marker -1 
    - Packer Reception Rate Trial 
    - Packet Reception Rate Current 
    - SyncTimestamp 
    - System Timestamp 
      - Cal and uncal 
    - Timestamp 
      - Cal and uncal
- We don’t have to use the uncal data, only cal data
- Total columns: 5 sensors, 5 * 33 columns + 2 extra columns (Annotation Level and Annotation Pulse)



Please refer to these publications for more details:
```
Singh, A., Le, B.T., Nguyen, T.L., Whelan, D., O’Reilly, M., Caulfield, B. and Ifrim, G., 2021, February. 
Interpretable classification of human exercise videos through pose estimation and multivariate time series analysis. 
In International Workshop on Health Intelligence (pp. 181-199). Cham: Springer International Publishing.
https://doi.org/10.1007/978-3-030-93080-6_14

Singh, A., Bevilacqua, A., Nguyen, T.L., Hu, F., McGuinness, K., O’Reilly, M., Whelan, D., Caulfield, B. and Ifrim, 
G., 2023. Fast and robust video-based exercise classification via body pose tracking and scalable multivariate 
time series classifiers. Data Mining and Knowledge Discovery, 37(2), pp.873-912.
https://doi.org/10.1007/s10618-022-00895-4

Singh, A., Bevilacqua, A., Aderinola, T.B., Nguyen, T.L., Whelan, D., O’Reilly, M., Caulfield, B. and Ifrim, G., 2023, 
September. An Examination of Wearable Sensors and Video Data Capture for Human Exercise Classification. 
In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 312-329). Cham: Springer 
Nature Switzerland.
https://doi.org/10.1007/978-3-031-43427-3_19
```


