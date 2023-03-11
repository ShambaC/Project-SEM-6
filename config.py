import os
from pickle import Pickler

class ModelConfig() :
    def __init__(self) :
        self.model_path = "Models/Handwriting_recognition"
        self.vocab = ''
        self.height = 32
        self.width = 128
        self.max_text_length = 0
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.train_epochs = 100
        self.train_workers = 20
        self.validation_split = 0.9

    # Save config as yaml file
    def save(self) :
        os.makedirs(self.model_path)
        file_name = f"{self.model_path}/configs.meow"
        
        with open(file_name, 'wb') as wf :
            Pickler(wf).dump(self)