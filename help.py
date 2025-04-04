# from tkinter import *
# from tkinter import ttk
# from PIL import Image, ImageTk  # pip install pillow
# from tkinter import messagebox
# import mysql.connector  #pip install mysql-connector-python
# import cv2  # pip install opencv-python
# import os
# import numpy as np

# class Help:
#     def __init__(self, root):
#         self.root = root
#         self.root.geometry("1530x790+0+0")
#         self.root.title("Face Recognition System")
        
#         #Title 
#         title_lbl=Label(self.root,text="Help Desk",font=("times new roman",35,"bold"),bg="white",fg="blue")
#         title_lbl.place(x=0,y=0,width=1530,height=45)
        
#         #image top
#         img_top = Image.open(r"images/attendance-management.jpg")
#         img_top = img_top.resize((1530,720),Image.LANCZOS)
#         self.photoimg_top = ImageTk.PhotoImage(img_top)

#         f_lbl = Label(self.root, image=self.photoimg_top)
#         f_lbl.place(x=0, y=50, width=1530, height=720)
        
#         dev_label = Label(f_lbl, text="Email : studiesbav.asifkhan@gmail.com  ", font=(
#             "times new roman", 20, "bold"), fg="blue")
#         dev_label.place(x=550,y=220)
        

# if __name__ == "__main__":
#     root = Tk()
#     obj = Help(root)
#     root.mainloop()
           