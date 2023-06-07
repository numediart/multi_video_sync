# Deep learning-based stereo camera multi-video synchronization

This repository is the official implementation of [Deep learning-based stereo camera multi-video synchronization](https://arxiv.org/abs/2303.12916) presented at ICASSP 2023.


## Requirements

To run this project, you will need the following dependencies:

- tensorflow
- tensorflow.keras
- tensorflow.data
- keras.util
- cv2
- numpy

Please note that the exact versions of these dependencies may vary. It is recommended to create a virtual environment and install the dependencies within it to avoid conflicts with other projects.

In the near future, a requirements.txt file will be added to the repository, which will provide a standardized way to install the necessary dependencies. This file will include the specific versions of the dependencies required for this project. Stay tuned for updates.

If you encounter any issues or have any questions, please don't hesitate to reach out to us.


## Data Preprocessing (Pre-processing only for model testing, not for training)
Before evaluate the AI model (this pre-processing is automatically performed during training), it is important to pre-process the data. This involves preparing databases of matching/non-matching images, as well as synchronized and desynchronized stereo videos. 

### 1 - Matching/non-matching images processing
This command creates a dataset of matching and non-matching image pairs, and the flow version. The following arguments can be used for data preprocessing:

- `--dbPath` : This argument specifies the path to the database of stereo images that will be processed. Make sure to provide the correct path to the database directory containing the necessary data.

- `--processImages` : By default, this argument is set to `1`, which means both the images and the flow will be processed. However, in case of errors (errors appeared frequently during the creation of the flows under Windows) or when you only want to process the flow component, you can set this argument to `0`.

To initiate the data preprocessing, run the preprocessing script with the desired arguments. For example:
```images processing
python preprocess_data.py --dbPath="/path/to/database" --processImages=1
```

### 2 - Stereo video processing

In order to generate videos for the AI model, a video creation process is required. This involves processing the previously created dataset and specifying the type of videos to be created. The following arguments can be used to create videos:

- `--dbPath` : This argument specifies the path to the previous dataset created with the Matching/non-matching images processing script. Make sure to provide the correct path to the dataset directory containing the necessary data.

- `--type` : By default, this argument is set to `flows`, which means that video sequences are created using optical flow frames. However, if you want to create videos from image frames, you can set this argument to `images`.

- `--pathReferences` : This argument should be set to the path of a numpy array of video references if you have already created them. Otherwise, leave it empty and the video creation process will generate the necessary references.

- `--numberVideos` : This argument determines the number of videos that will be created. The default value is set to 3000, but feel free to adjust it based on your requirements.

To initiate the video creation process, run the video creation script with the desired arguments. For example:
```video processing
python create_videos.py --dbPath="/path/to/dataset" --type="flows" --numberVideos=3000
```

For information, you can modify the video duration in the code on lines 14 and 15 of the tool/process_video.py file.

## Training

To train the AI model, you need to specify the dataset and the type of data to be used for training. The training script will handle the data preprocessing automatically, so there is no need for separate (previous section) preprocessing steps. The following arguments can be used for model training:

- `--dbPath` : This argument specifies the path to the dataset of stereo images that will be used to train the models. Make sure to provide the correct path to the dataset directory containing the necessary data.

- `--type` : By default, this argument is set to `flows`, which means the model will be trained using video sequences of optical flow frames. However, if you want to train the model using image frames, you can set this argument to `images`.

- `--pathReferenceList` : This argument should be set to the path of a numpy array of video references if you have already created them during the video creation process. This is an optional argument and can be left empty if the video references haven't been created yet.

- `--numberVideo` : This argument determines the number of videos created for training the DenseDelay model. The default value is set to 40000, but feel free to adjust it based on your requirements.

To start the model training process, run the training script with the desired arguments. For example:
```training
python train_model.py --dbPath="/path/to/dataset" --type="flows" --numberVideo=50000
```


## Evaluation

To evaluate the performance of the AI model, you can perform testing on a separate dataset. Before executing the testing script, **ensure that the dataset has been preprocessed** using the preprocessing steps mentioned earlier.

The following arguments can be used for model testing:

- `--dbPath` : This argument specifies the path to the dataset that will be tested. Make sure to provide the correct path to the dataset directory containing the preprocessed data.

- `--weights` : This argument specifies the weights used to evaluate the DenseDelay model. By default, it is set to the path "model\DenseDelay\weight". Ensure that you have the correct weights file or update the path accordingly.

- `--type` : By default, this argument is set to `flows`, which means the evaluation will be performed on video sequences representing optical flow. However, if you want to evaluate the model using image frames, you can set this argument to `images`. It is important to select weights trained with the same type of data.

To start the model testing script, run the script with the desired arguments. For example:
```testing
python test_model.py --dbPath="/path/to/dataset" --weights"/path/to/weights" --type="flows"
```

## Dataset
For the dataset, contact the authors of the paper.
