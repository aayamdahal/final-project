from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import glob
from os import path
import os
from functools import partial

from utils import identify_and_evaluate


class PaintCanvas:
    def __init__(self,WINDOW):
        self.WINDOW = WINDOW
        self.old_x = None
        self.old_y = None
        self.penWidth = 10
        self.canvas_width = 800
        self.canvas_height = 256
        self.penColor = 'black'
        self.fg_color = (0,0,0)
        self.bg_color = (255,255,255)
        self.createWidgets()

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y,
                               width=self.penWidth, fill=self.penColor,
                               capstyle=ROUND, smooth=True)
            self.draw.line([self.old_x, self.old_y, e.x, e.y], self.fg_color,
                           width=4)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):
        # reseting or cleaning the canvas
        self.old_x = None
        self.old_y = None

    def createWidgets(self):
        white = (255, 255, 255)

        x,y = 100, 100

        ShowButton(self.WINDOW, "Pen", x, y, self.selectPen)
        ShowButton(self.WINDOW, "Eraser", x + 100, y, self.selectEraser)
        ShowButton(self.WINDOW, "Clear", x + 200, y, self.selectClear)
        ShowButton(self.WINDOW, "Evaluate", x + 300, y, self.saveImage)

        self.c = Canvas(self.WINDOW, width=self.canvas_width,
                        height=self.canvas_height, bg='white')
        self.c.pack()
        self.c.place(x=100, y=y + 48)

        # draw the line
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.image1 = Image.new("RGB", (self.canvas_width, self.canvas_height),
                                self.bg_color)
        self.draw = ImageDraw.Draw(self.image1)

    def selectClear(self):
        self.c.delete("all")
        self.draw.line([0, 0, 1200, 1200], self.bg_color, width=1200)

    def selectPen(self):
        self.penWidth = 10
        self.penColor = "black"
        self.fg_color = (0, 0, 0)

    def selectEraser(self):
        self.penWidth = 30
        self.penColor = "white"
        self.fg_color = (255, 255, 255)

    def saveImage(self):
        images_path = os.getcwd() + '\\data\\'
        filename = images_path + "answer.jpg"

        self.image1.save(filename)

        answer = identify_and_evaluate(filename)
        confirm = messagebox.askquestion('Confirm', 'Answer: ' + str(answer))
        self.selectClear()
        if confirm == 'no':
            return

def ShowButton(WINDOW,text,X,Y,Command):
    panel = Button(WINDOW, text=text, style='TButton', command=Command)
    panel.pack()
    panel.place(x=X, y=Y)


WINDOW = Tk()
WINDOW.geometry('+200+50')
WINDOW.geometry('1000x600')
WINDOW.title("Handwritten expression evaluator")
style = Style()
style.configure('W.TButton', font=('calibri', 15, 'bold'), foreground='black',
                borderwidth='4')
style.configure('TButton', font=('calibri', 13, 'bold'), foreground='black',
                borderwidth='4')

def menu():
    PaintCanvas(WINDOW)

menu()
WINDOW.mainloop()
