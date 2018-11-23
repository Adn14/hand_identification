from tkinter.ttk import *
from tkinter import *
from tkinter.filedialog import *
from PIL import Image,ImageTk,ImageFilter,ImageOps
import tkinter as tk
import tkinter,tkinter.filedialog
import cv2
import os
import capture
import pred_fing
import processPred
import training
import numpy as np

global last_frame, cap, file, im
cap = cv2.VideoCapture(0)

def pred():
    processPred.run()
    hasil = pred_fing.predict()
    text4.delete('1.0', END)
    text4.insert(INSERT, "hasil prediksi : " + str(hasil))
    file = "jari.jpg"
    im = Image.open(file)
    im = im.resize((534,300))
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(root, image = tkimage)
    myvar.image = tkimage
    myvar.configure(image=tkimage, borderwidth=2, relief="groove")
    myvar.place(x=10, y=15, width=534, height=300)
    
    
    
def latih(error, epos):
    if error == 0 or epos == 0:
        acc, err, epos = training.train(0.1, 10)
    else:
        acc, err, epos = training.train(error, epos)
    
    text.insert(INSERT,"akurasi : "+str(acc)+ "\nTotal Epos : "+ str(epos) )
    return epos, err

def open_file():
    
    #print(file)
    return file

if __name__ == '__main__':
    root=Tk()
    lmain = tk.Label(master=root)
    lmain.grid(column=0, rowspan=4, padx=5, pady=5)
    root.title("Final Exam Project")
    root.geometry("875x500")
    root.resizable(width=False, height=False)
    
    epos = IntVar()
    error = DoubleVar()

    labelc = Label(text="Take a Picture, please!!", borderwidth=2, relief="groove")
    labelc.place(x=10, y=15, width=534, height=300)

    btn = Button(text="Take a Picture", command=lambda:capture.capt_frame)
    btn.place(x=10, y=325, width=444, height=40)

    btn1 = Button(text="Browse Image", command=lambda:open_file)
    btn1.place(x=459, y=325, width=85, height=40)

    btn2 = Button(text="Training", command=lambda:latih(error.get(), epos.get()))
    btn2.place(x=565, y=15, width=60, height=60)
    text = Text()
    text.place(x=630, y=15, width=235, height=60)

    label = Label(text="Accuration")
    label.place(x=565, y=85)
    text1 = Text()
    text1.place(x=565, y=105, width=300, height=30)

    label1 = Label(text="Error")
    label1.place(x=565, y=140)
    text2 = Entry(textvariable = error)
    text2.place(x=565, y=160, width=300, height=30)

    label1 = Label(text="Epoch")
    label1.place(x=565, y=195)
    text3 = Entry(textvariable = epos)
    text3.place(x=565, y=215, width=300, height=30)

    btn3 = Button(text="Predict", command=lambda:pred())
    btn3.place(x=565, y=265, width=100, height=30)
    
    text4 = Text()
    text4.place(x=670, y=265, width=200, height=100)
   

    label1 = Label(justify="right",text="Created by:\nAdn Agung Rochman Arifin\nAlbert Wahyudi Nur\nImmanuel Olive Djaja Putra\nYohanes Hans Kristian")
    label1.place(x=715, y=420)

    root.mainloop()
    cap.release()