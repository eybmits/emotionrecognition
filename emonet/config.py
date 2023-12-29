import os
 
# initialize the path to the root folder where the dataset resides and the
# path to the train and test directory


DATASET_FOLDER = f'data'
TRAIN_DIRECTORY = os.path.join(DATASET_FOLDER, "train")
TEST_DIRECTORY = os.path.join(DATASET_FOLDER, "test")




# initialize the amount of samples to use for training and validation
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10
 
# specify the batch size, total number of epochs and the learning rate
BATCH_SIZE = 32
NUM_OF_EPOCHS = 100
LR = 1e-4
