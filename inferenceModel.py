import cv2
import numpy as np
from itertools import groupby
import tensorflow as tf
from config import ModelConfig
from CustomTF import CTCloss, CWERMetric

def ctc_decoder(predictions, chars) :

    argmax_preds = np.argmax(predictions, axis= -1)
                             
    # grouped_preds = [[k for k, _ in groupby(preds)] for preds in argmax_preds]
    grouped_preds = [[preds for preds in argmax_preds]]

    texts = ["".join([chars[k] for k in group if k < len(chars)]) for group in grouped_preds]

    return texts

unixTime = 1678680123
configFilePath = f"Models/Handwriting_recognition/{unixTime}/configs.meow"
configs = ModelConfig().load(configFilePath)

model = tf.keras.models.load_model(f"{configs.model_path}/model.meow", compile= False)

def recog(img, model, config) :
    img = cv2.resize(img, (128, 32), interpolation= cv2.INTER_AREA)

    preds = model.predict(np.array([img]))[0]
    text = ctc_decoder(preds, config.vocab)[0]

    return text

if __name__ == "__main__" :
    img = cv2.imread("Datasets/words/a01/a01-000u/a01-000u-00-01.png")
    pred_text = recog(img, model, configs)
    print(pred_text)
    img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4))
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()