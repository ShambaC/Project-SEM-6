# Project SEM 6
Handwritten text recognition

## What ?
Handwritten text recognition using Long Short Term Memory implementation of RNN. Honestly I don't know what's happening here.
The [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) is used to traing the network. Specifically the words dataset.

## How to ?
Clone the repo and then do the following.

### Get the IAM Handwriting Dataset :
- Make an [account](https://fki.tic.heia-fr.ch/register) on the database website.
- Download the [words archive](https://fki.tic.heia-fr.ch/DBs/iamDB/data/words.tgz)
- Extract the contents to the following directory structure on the root of this project : `Datasets/words`
    - So the resulting folder structure should be : 
        ```
        Datasets
            |_words
                |_a01
                    |_a01-000u
                        |_images.png
                    |_a01-000x
                    |_...
                |_a02
                |_...
        ```
- Now you need the labels for these images.
- Download the [ascii archive](https://fki.tic.heia-fr.ch/DBs/iamDB/data/ascii.tgz)
- Extract only the words.txt file to the `Datasets` folder.

### Install the dependencies
- install the requirements.
- Do in terminal : `pip install -r requirements.txt`
- The required packages are :
    - numpy : 1.22.4
    - opencv_python : 4.6.0.66
    - pandas : 1.3.2
    - Pillow : 9.4.0
    - tensorflow : 2.10.0
    - tqdm : 4.62.3

### Edit the configurations
- Open the `config.py` file.
- Make whatever changes you want.
- Mainly you might want to make changes to the following variables : 
    - batch_size
    - learning_rate
    - train_epochs
    - train_workers
    - validation_split

### Train the model
- Run the `train.py` script to train the model.
- No need for editing anything in this file.
- I trained the model with the following parameters :
    - learning rate : 0.0005
    - validation split : 0.9
    - batch size = 16
    - epochs = 100
- I trained it with my GTX 1650. It used 2132 MB of GPU memory. Usage was around 8-10 %. Took around 140-150 seconds each epoch. Took about 4 hours for all 100 epochs.
- CPU usage was around 50%. My CPU is Ryzen 7 3750H.
- RAM usage was around 2 GBs.
- You can visualise the training using tensorboard. Run `tensorboard --logdir = path_to_logs` in terminal to start the server.
- The logs are located at the following folder : `Models/Handwriting_recognition/{timestamp}/logs`
- After training a bunch of files are generated. The only important files are `model.meow` and `configs.meow`.

### Run the model
- You can run the `inferenceModel.py` to check with only one image from the dataset itself.
- OR use the Paint GUI made with tkinter by running the `tkRecog.py`.
# IN BOTH CASES MAKE SURE TO EDIT THE unixTime VARIABLE WITH YOUR MODEL'S FOLDER

# Thank You
Totally not copied code.