from inferenceModel import ctc_decoder
from config import ModelConfig
import tensorflow as tf
import cv2
import numpy as np

unixTime = 1678680123
configFilePath = f"Models/Handwriting_recognition/{unixTime}/configs.meow"
configs = ModelConfig().load(configFilePath)
model = tf.keras.models.load_model(f"{configs.model_path}/model.meow", compile= False)

def recog(img) :
    img = cv2.resize(img, (128, 32), interpolation= cv2.INTER_AREA)

    preds = model.predict(np.array([img]))[0]
    text = ctc_decoder(preds, configs.vocab)[0]

    return text

# PAINT
from tkinter import *
from tkinter import messagebox
import PIL.ImageGrab as ImageGrab

class Draw() :
    def __init__(self, root) :
        self.root = root
        self.root.title("Paint")
        self.root.geometry("1124x340")
        self.root.configure(background = "black")
        self.root.resizable(0, 0)

        self.pointer = "black"

        # Reset Button to clear the entire screen
        self.clear_screen = Button(self.root, text = "Clear Screen", bd = 4, bg = 'white', command = lambda : self.background.delete('all'), width = 9, relief = RIDGE)
        self.clear_screen.place(x = 0, y = 127)
 
        # Button to recognise the drawn text
        self.rec_btn = Button(self.root, text = "Recognise", bd = 4, bg = 'white', command = self.rec_drawing, width = 9, relief = RIDGE)
        self.rec_btn.place(x = 0, y = 157)

        #Defining a background color for the Canvas
        self.background = Canvas(self.root, bg = 'white', bd = 5, relief = FLAT, height = 256, width = 1024)
        self.background.place(x = 80, y = 40)
 
        #Bind the background Canvas with mouse click
        self.background.bind("<B1-Motion>",self.paint)

    def paint(self, event) :
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)

        self.background.create_oval(x1, y1, x2, y2, fill = self.pointer, outline = self.pointer, width = 20)

    def rec_drawing(self) :
        try :
            x = self.root.winfo_rootx() + self.background.winfo_x()
            y = self.root.winfo_rooty() + self.background.winfo_y()

            x1 = x + self.background.winfo_width()
            y1 = y + self.background.winfo_height()
            img = ImageGrab.grab().crop((x + 7, y + 7, x1 - 7, y1 - 7))

            text = recog(np.array(img))
            messagebox.showinfo(title= 'Result', message= f"The word is {text}")
        except :
            print("Error while screenshoting or recognising")

root = Tk()
p = Draw(root)
root.mainloop()