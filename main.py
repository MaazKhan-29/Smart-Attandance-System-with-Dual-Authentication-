# from tkinter import *
# from tkinter import ttk
# import tkinter.messagebox
# from PIL import Image, ImageTk  # pip install pillow
# from student import Student
# import os
# from train import Train
# from face_recognition_utils import Face_Recognition
# from attandance import Attandance
# from developer import Developer
# from help import Help
# import tkinter
# from time import strftime
# from datetime import datetime


# class face_recognition_system:
#     def __init__(self, root):
#         self.root = root
#         self.root.geometry("1530x790+0+0")
#         self.root.title("Face Recognition System")

#         # First image
#         img = Image.open("images/login.jpg")
#         img = img.resize((500, 130),Image.LANCZOS)
#         self.photoimg = ImageTk.PhotoImage(img)

#         f_lbl = Label(self.root, image=self.photoimg)
#         f_lbl.place(x=0, y=0, width=500, height=130)


#         #Second image
#         img1 = Image.open(r"images/attendance-management.jpg")
#         img1 = img1.resize((500, 130),Image.LANCZOS)
#         self.photoimg1 = ImageTk.PhotoImage(img1)

#         f_lbl = Label(self.root, image=self.photoimg1)
#         f_lbl.place(x=500, y=0, width=500, height=130)


#         #Third image
#         img2 = Image.open(r"images/attendance-management.jpg")
#         img2 = img2.resize((500, 130),Image.LANCZOS)
#         self.photoimg2 = ImageTk.PhotoImage(img2)

#         f_lbl = Label(self.root, image=self.photoimg2)
#         f_lbl.place(x=1000, y=0, width=500, height=130)


#         #Background image
#         img3 = Image.open(r"images/login.jpg")
#         img3 = img3.resize((1530,710),Image.LANCZOS)
#         self.photoimg3 = ImageTk.PhotoImage(img3)

#         bg_img = Label(self.root, image=self.photoimg3)
#         bg_img.place(x=0, y=130, width=1530, height=710)

#         #Title 
#         title_lbl=Label(bg_img,text="FACE RECOGNITION ATTANDANCE SYSTEM",font=("times new roman",35,"bold"),bg="white",fg="red")
#         title_lbl.place(x=0,y=0,width=1530,height=45)
        
#         def time():
#             string=strftime('%H:%M:%S %p')
#             lbl.config(text = string)
#             lbl.after(1000, time)
        
#         lbl = Label(title_lbl,font=("times new roman",14,"bold"),background="white",foreground="blue")
#         lbl.place(x=10,y=0,width=110,height=50)
#         time()

#         #student detail button1
#         img4 = Image.open(r"images/attendance-management.jpg")
#         img4 = img2.resize((220, 200),Image.LANCZOS)
#         self.photoimg4 = ImageTk.PhotoImage(img4)

#         b1 = Button(bg_img, image=self.photoimg4,command=self.student_details,cursor="hand2")
#         b1.place(x=120, y=100, width=220, height=200)

#         b1_1 = Button(bg_img, text="Student details",command=self.student_details,cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
#         b1_1.place(x=120, y=280, width=220, height=40)


#         #detect face button
#         img5 = Image.open(r"images/face.jpg")
#         img5 = img2.resize((220, 200),Image.LANCZOS)
#         self.photoimg5 = ImageTk.PhotoImage(img5)

#         b1 = Button(bg_img,command=self.face_data, image=self.photoimg5,cursor="hand2")
#         b1.place(x=440, y=100, width=220, height=200)

#         b1_1 = Button(bg_img,command=self.face_data,  text="Face detection",cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
#         b1_1.place(x=440, y=280, width=220, height=40)

#         #attandance button
#         img6 = Image.open(r"images/attendance-management.jpg")
#         img6 = img2.resize((220, 200),Image.LANCZOS)
#         self.photoimg6 = ImageTk.PhotoImage(img6)

#         b1 = Button(bg_img,command=self.attandance_func, image=self.photoimg6,cursor="hand2")
#         b1.place(x=760, y=100, width=220, height=200)

#         b1_1 = Button(bg_img,command=self.attandance_func, text="Attandance",cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
#         b1_1.place(x=760, y=280, width=220, height=40)

#         #help button
#         img7 = Image.open(r"images/attendance-management.jpg")
#         img7 = img2.resize((220, 200),Image.LANCZOS)
#         self.photoimg7 = ImageTk.PhotoImage(img7)

#         b1 = Button(bg_img,command=self.help_desk, image=self.photoimg7,cursor="hand2")
#         b1.place(x=1080, y=100, width=220, height=200)

#         b1_1 = Button(bg_img,command=self.help_desk, text="Help Desk",cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
#         b1_1.place(x=1080, y=280, width=220, height=40)


#         #train face button
#         img8 = Image.open(r"images/attendance-management.jpg")
#         img8 = img2.resize((220, 200),Image.LANCZOS)
#         self.photoimg8 = ImageTk.PhotoImage(img8)

#         b1 = Button(bg_img,command=self.train_data,image=self.photoimg8,cursor="hand2")
#         b1.place(x=120, y=380, width=220, height=200)

#         b1_1 = Button(bg_img, text="Train Data",command=self.train_data ,cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
#         b1_1.place(x=120, y=560, width=220, height=40)

#         #photos button
#         img9 = Image.open(r"images/attendance-management.jpg")
#         img9 = img2.resize((220, 200),Image.LANCZOS)
#         self.photoimg9 = ImageTk.PhotoImage(img7)

#         b1 = Button(bg_img, image=self.photoimg9,cursor="hand2",command=self.open_img)
#         b1.place(x=440, y=380, width=220, height=200)

#         b1_1 = Button(bg_img, text="Stored Photos",cursor="hand2",command=self.open_img,font=("times new roman",20,"bold"),bg="darkblue",fg="white")
#         b1_1.place(x=440, y=560, width=220, height=40)

#         #developer
#         img10 = Image.open(r"images/attendance-management.jpg")
#         img10 = img2.resize((220, 200),Image.LANCZOS)
#         self.photoimg10 = ImageTk.PhotoImage(img10)

#         b1 = Button(bg_img, command=self.developer_data, image=self.photoimg10,cursor="hand2")
#         b1.place(x=760, y=380, width=220, height=200)

#         b1_1 = Button(bg_img, command=self.developer_data, text="Developer",cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
#         b1_1.place(x=760, y=560, width=220, height=40)

#         #exit
#         img11 = Image.open(r"images/attendance-management.jpg")
#         img11 = img2.resize((220, 200),Image.LANCZOS)
#         self.photoimg11 = ImageTk.PhotoImage(img11)

#         b1 = Button(bg_img, command=self.iExit, image=self.photoimg11,cursor="hand2")
#         b1.place(x=1080, y=380, width=220, height=200)

#         b1_1 = Button(bg_img, command=self.iExit,text="EXIT",cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
#         b1_1.place(x=1080, y=560, width=220, height=40)

#     # opens data folder by clicking photos button
#     def open_img(self):
#         os.startfile("Data")
        
#     def iExit(self):
#         self.iExit=tkinter.messagebox.askyesno("Face Recognition","Are you Sure to Exit this project",parent=self.root)
#         if self.iExit>0:
#             self.root.destroy()
#         else:
#             return    
        
#     #Function buttons

#     def student_details(self):
#         self.new_window=Toplevel(self.root)
#         self.app=Student(self.new_window)
        
#     def train_data(self):
#         self.new_window=Toplevel(self.root)
#         self.app=Train(self.new_window)
        
#     def face_data(self):
#         self.new_window=Toplevel(self.root)
#         self.app=Face_Recognition(self.new_window)
        
#     def attandance_func(self):
#         self.new_window=Toplevel(self.root)
#         self.app=Attandance (self.new_window)
        
#     def developer_data(self):
#         self.new_window=Toplevel(self.root)
#         self.app=Developer (self.new_window)
        
#     def help_desk(self):
#         self.new_window=Toplevel(self.root)
#         self.app=Help (self.new_window)







# if __name__ == "__main__":
#     root = Tk()
#     obj = face_recognition_system(root)
#     root.mainloop()
    
    
    
    
