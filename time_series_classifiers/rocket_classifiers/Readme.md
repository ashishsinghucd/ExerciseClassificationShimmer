The code in the file `rocket.py` is used to classify the generated train/test data from previous section. It loops
over 3 seeds values and outputs the classification results. 

```
INPUT_DATA_PATH = input data path
BASE_PATH= base path
EXERCISE = type of exercise
SEED_VALUES= seed values for the number of splits
OUTPUT_PATH =  output path
DATA_TYPE = data type
GENDER_INFO= demographic information file
```

The code of this file can be modified for other ROCKET based transforms such as MiniROCKET or MultiROCKET.

The code in file `rocket_combined_modality.py` is used to create an ensemble model by concatenating data from
both video and sensors. 
The code in file `rocket_combined_modality.py` is used to create an ensemble model where in two independent
models are trained on video and sensor data separately. The predictions during testing are combined using 
average probabilities.