import tensorflow as tf

from config import ModelConfig
from model import train_model
from CustomTF import CTCloss, CWERMetric

from tqdm import tqdm
import os

# Preprocess
dataset = []
vocab = set()
max_len = 0


words = open("Datasets/words.txt", "r").readlines()
for line in tqdm(words) :

    # Skip lines that start with a hashtag as those are comments
    if line.startswith('#') :
        continue

    """
    Lines store information regarding the pictures in the dataset.
    They have a particular syntax of storing data

    format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A

    a01-000u-00-00      -> word id
    a01                 -> top level folder
    a01-000u            -> next level, contains the images
    a01-000u-'00-00'    -> quoted part is the image id.
    
    ok                  -> Result of the image segmentation
                                >ok = can be segmented properly
                                >err = segmentation can be bad
    154                 -> graylevel threshold to binarize the image
    1                   -> number of components for this word (seems like none of the lines has this)
    408 768 27 51       -> bounding box for this word in x, y, w, h format
    AT                  -> Grammatical tag based on IAM tagset
    A                   -> Transcription for the word (label)
    """    
    line_split = line.split(' ')

    if line_split[1] == 'err' :
        continue

    folder1 = line_split[0][:3]
    folder2 = line_split[0][:8]
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip('\n')

    path = f"Datasets/words/{folder1}/{folder2}/{file_name}"
    if not os.path.exists(path) :
        continue

    dataset.append([path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

# Create configurations object to store data
configs = ModelConfig()

# Save the vocab and max text length in config
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Modify dataset for tensorflow

import cv2
import numpy as np
import copy

for tups in dataset :
    img = cv2.imread(tups[0])
    img = cv2.resize(img, (configs.width, configs.height), interpolation= cv2.INTER_AREA)
    cv2.imwrite(tups[0], img)

    labIndex = np.array([configs.vocab.index(l) for l in tups[1] if l in configs.vocab])
    labPad = np.pad(labIndex, (0, configs.max_text_length - len(labIndex)), 'constant', constant_values=len(configs.vocab))

    tups[1] = labPad

# Validation split
np.random.shuffle(dataset)
train_data = copy.deepcopy(dataset)
val_data = copy.deepcopy(dataset)

train_data = train_data[ : int(len(dataset) * configs.validation_split)]
val_data = val_data[int(len(dataset) * configs.validation_split) : ]

# Create model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab)
)

# Compile model and print summary
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = configs.learning_rate),
    loss = CTCloss(),
    metrics = [CWERMetric(padding_token = len(configs.vocab))]
)
model.summary(line_length = 110)

# Callbacks
earlystopper = tf.keras.callbacks.EarlyStopping(monitor= 'val_CER', patience= '20', verbose= 1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{configs.model_path}/model.h5", monitor= 'val_CER', verbose= 1, save_best_only= True, mode= 'min')
tb_callback = tf.keras.callbacks.TensorBoard(f"{configs.model_path}/logs", update_freq= 1)
reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor= 'val_CER', factor= 0.9, min_delta= 1e-10, patience= 10, verbose= 1, mode= 'auto')

# Train
model.fit(
    train_data,
    validation_data = val_data,
    epochs = configs.train_epochs,
    callbacks = [earlystopper, checkpoint, tb_callback, reduceLROnPlat],
    workers = configs.train_workers
)

# saving datasets as csv
import pandas as pd

path = f"{configs.model_path}/train.csv"
df = pd.DataFrame(train_data)
df.to_csv(path, index= False)

path = f"{configs.model_path}/val.csv"
df = pd.DataFrame(val_data)
df.to_csv(path, index= False)