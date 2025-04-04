from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk  # Ensure Pillow is installed: pip install pillow
from tkinter import messagebox
import mysql.connector
from Student import Student
import os
#from train import Train
from face_recognition_utils import Face_Recognition
from attandance import Attandance
# from developer import Developer
# from help import Help
import tkinter
from time import strftime
from datetime import datetime


def main():
    win=Tk()
    app=Login_Window(win)
    win.mainloop()
    
class Login_Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.root.geometry("1550x800+0+0")
       
        
        img0 = Image.open(r"images/Loginbg.jpg")
        img0 = img0.resize((1550, 800),  Image.LANCZOS)
        self.photoimage0 = ImageTk.PhotoImage(img0)
        lblimg0 = Label(self.root, image=self.photoimage0, borderwidth=0)
        lblimg0.place(x=0, y=0, width=1550, height=800)
# 
        # Login Frame
        frame = Frame(self.root, bg="black")
        frame.place(x=500, y=170, width=360, height=450) #610 #340

        # User Icon
        img1 = Image.open(r"images/ico.png")
        img1 = img1.resize((100, 100),  Image.LANCZOS)
        self.photoimage1 = ImageTk.PhotoImage(img1)
        lblimg1 = Label(frame, image=self.photoimage1, bg="black", borderwidth=0)
        lblimg1.place(x=120, y=10, width=100, height=100)

        # Title
        get_str = Label(frame, text="Get Started", font=("times new roman", 20, "bold"), fg="white", bg="black")
        get_str.place(x=95, y=100)

        # Username Label and Entry
        lbluser = Label(frame, text="Username", font=("times new roman", 15, "bold"), fg="white", bg="black")
        lbluser.place(x=55, y=150)
        self.var_email = ttk.Entry(frame, font=("times new roman", 15, "bold"))
        self.var_email.place(x=60, y=180, width=270)

        # Password Label and Entry
        lblpass = Label(frame, text="Password", font=("times new roman", 15, "bold"), fg="white", bg="black")
        lblpass.place(x=55, y=222)
        self.var_pass = ttk.Entry(frame, font=("times new roman", 15, "bold"), show='*')
        self.var_pass.place(x=60, y=250, width=270)

        # Username Icon
        img2 = Image.open(r"images/ico.png")
        img2 = img2.resize((35, 35),  Image.LANCZOS)
        self.photoimage2 = ImageTk.PhotoImage(img2)
        lblimg2 = Label(frame, image=self.photoimage2, bg="black", borderwidth=0)
        lblimg2.place(x=20, y=180, width=25, height=25)

        # Password Icon
        img3 = Image.open(r"images/key2.png")
        img3 = img3.resize((35, 35),  Image.LANCZOS)
        self.photoimage3 = ImageTk.PhotoImage(img3)
        lblimg3 = Label(frame, image=self.photoimage3, bg="black", borderwidth=0)
        lblimg3.place(x=20, y=250, width=25, height=25)

        # Login Button
        loginbtn = Button(frame, command=self.login, text="Login", font=("times new roman", 15, "bold"), bd=3, relief=RIDGE, fg="white", bg="blue", activeforeground="white", activebackground="darkblue")
        loginbtn.place(x=120, y=300, width=120, height=35)

        # Register Button
        registerbtn = Button(frame, command=self.register_window, text="New User Register",bd=3, font=("times new roman", 13, "bold"), borderwidth=0, fg="white", bg="blue", activeforeground="white", activebackground="black")
        registerbtn.place(x=30, y=370, width=150,height=35)

        # Forget Password Button
        forgetbtn = Button(frame, text="Forget Password", font=("times new roman", 13, "bold"),bd=3, borderwidth=0, fg="white", bg="blue", activeforeground="white", activebackground="black")
        forgetbtn.place(x=195, y=370, width=140,height=35)
    
    def register_window(self):
        self.new_window=Toplevel(self.root)
        self.app=Register(self.new_window)
    
 
    
    def login(self):
        if self.var_email.get() == "" or self.var_pass.get() == "":
            messagebox.showerror("Error", "All fields are required")
        elif self.var_email.get() == "ak" and self.var_pass.get() == "pk":
            messagebox.showinfo("Success", "Welcome to codewithkiran channel, please subscribe to my channel")
        else:
            conn = mysql.connector.connect(host="localhost", user="root",  password="Maaz%2006", database="maazdb")
            my_cursor = conn.cursor()

            my_cursor.execute("select * from register where email=%s and password=%s", (
            self.var_email.get(),
            self.var_pass.get() 
                                 ))
            row = my_cursor.fetchone()

            if row==None:
                messagebox.showerror("Error", "Invalid Username & password")
            else:
                open_main = messagebox.askyesno("YesNo", "Access only admin")
                if open_main > 0:
                    
                    self.new_window = Toplevel(self.root)
                    self.app =  face_recognition_system(self.new_window)
                else:
                    if not open_main:
                        return
            conn.commit()
            conn.close() 
            
             

class Register:
    def __init__(self, root):
        self.root = root
        self.root.title("Register")
        self.root.geometry("1600x900+0+0")
        
        #=============== Variablees =====================
        self.var_fname=StringVar()
        self.var_lname=StringVar()
        self.var_contact=StringVar()
        self.var_email=StringVar()
        self.var_securityQ=StringVar()
        self.var_securityA=StringVar()
        self.var_pass=StringVar()
        self.var_confpass=StringVar()

        """# Left Image
        self.bg1 = ImageTk.PhotoImage(file=r"images/attendance-management.jpg")
        left_lbl = Label(self.root, image=self.bg1)
        left_lbl.place(x=50, y=100, width=470, height=550)"""
        
        img00 = Image.open(r"images/Loginbg.jpg")
        img00 = img00.resize((1550, 800),  Image.LANCZOS)
        self.photoimage00 = ImageTk.PhotoImage(img00)
        lblimg0 = Label(self.root, image=self.photoimage00, borderwidth=0)
        lblimg0.place(x=0, y=0, width=1550, height=800)
        
        img99 = Image.open(r"images/eye.jpg")
        img99 = img99.resize((550, 600),  Image.LANCZOS)
        self.photoimage99 = ImageTk.PhotoImage(img99)
        lblimg0 = Label(self.root, image=self.photoimage99, borderwidth=0)
        lblimg0.place(x=230, y=100, width=400, height=550)
        
        # Main Frame
        frame = Frame(self.root, bg="black")
        frame.place(x=520, y=100, width=800, height=550)
        
        register_lbl = Label(frame, text="REGISTER HERE", font=("times new roman", 35, "bold"), fg="blue", bg="black")
        register_lbl.place(x=180, y=20)
        
        # Labels and Entry Fields
        # Row 1
        fname = Label(frame, text="First Name", font=("times new roman", 15, "bold"), bg="black", fg="white")
        fname.place(x=100, y=100)
        self.fname_entry = ttk.Entry(frame,textvariable=self.var_fname, font=("times new roman", 15, "bold"))
        self.fname_entry.place(x=100, y=130, width=250)
        
        l_name = Label(frame, text="Last Name", font=("times new roman", 15, "bold"), bg="black", fg="white")
        l_name.place(x=450, y=100)
        self.txt_lname = ttk.Entry(frame,textvariable=self.var_lname, font=("times new roman", 15))
        self.txt_lname.place(x=450, y=130, width=250)
        
        # Row 2
        contact = Label(frame, text="Contact No", font=("times new roman", 15, "bold"), bg="black", fg="white")
        contact.place(x=100, y=170)
        self.txt_contact = ttk.Entry(frame,textvariable=self.var_contact, font=("times new roman", 15))
        self.txt_contact.place(x=100, y=200, width=250)
        
        var_email = Label(frame, text="Email", font=("times new roman", 15, "bold"), bg="black", fg="white")
        var_email.place(x=450, y=170)
        self.txt_email = ttk.Entry(frame,textvariable=self.var_email, font=("times new roman", 15))
        self.txt_email.place(x=450, y=200, width=250)
        
        # Row 3
        security_Q = Label(frame, text="Select Security Question", font=("times new roman", 15, "bold"), bg="black", fg="white")
        security_Q.place(x=100, y=240)
        
        self.combo_security_Q = ttk.Combobox(frame,textvariable=self.var_securityQ, font=("times new roman", 15, "bold"), state="readonly")
        self.combo_security_Q["values"] = ("Select", "Your Birth Place", "Your Girlfriend Name", "Your Pet Name")
        self.combo_security_Q.place(x=100, y=270, width=250)
        self.combo_security_Q.current(0)
        
        security_A = Label(frame, text="Security Answer", font=("times new roman", 15, "bold"), bg="black", fg="white")
        security_A.place(x=450, y=240)
        self.txt_security = ttk.Entry(frame,textvariable=self.var_securityA, font=("times new roman", 15))
        self.txt_security.place(x=450, y=270, width=250)
        
        # Row 4
        pswd = Label(frame, text="Password", font=("times new roman", 15, "bold"), bg="black", fg="white")
        pswd.place(x=100, y=310)
        self.txt_pswd = ttk.Entry(frame,textvariable=self.var_pass,font=("times new roman", 15))
        self.txt_pswd.place(x=100, y=340, width=250)
        
        confirm_pswd = Label(frame, text="Confirm Password", font=("times new roman", 15, "bold"), bg="black", fg="white")
        confirm_pswd.place(x=450, y=310)
        self.txt_confirm_pswd = ttk.Entry(frame,textvariable=self.var_confpass, font=("times new roman", 15))
        self.txt_confirm_pswd.place(x=450, y=340, width=250)
        
        # Checkbox
        self.var_check=IntVar()
        checkbtn = Checkbutton(frame,variable=self.var_check, text="I Agree The Terms & Conditions", font=("times new roman", 12, "bold"), bg="black", fg="white", onvalue=1, offvalue=0)
        checkbtn.place(x=100, y=380)
        
        # Buttons
        img = Image.open(r"images/login_btn.jpeg")
        img = img.resize((200, 55),  Image.LANCZOS)
        self.photoimage = ImageTk.PhotoImage(img)
        
        b1 = Button(frame, command=self.register_data, image=self.photoimage, borderwidth=0, cursor="hand2", font=("times new roman", 15, "bold"), fg="white")
        b1.place(x=100, y=450, width=200)
        
        # img1 = Image.open(r"images/attendance-management.jpg")
        # img1 = img1.resize((200, 55),  Image.LANCZOS)
        # self.photoimage1 = ImageTk.PhotoImage(img1)
        
        # b1 = Button(frame, image=self.photoimage1, borderwidth=0, cursor="hand2", font=("times new roman", 15, "bold"), fg="white")
        # b1.place(x=360, y=450, width=200)
        
    # =================Function declaration============================
  
    def register_data(self):
        if self.var_fname.get() == "" or self.var_email.get() == "" or self.var_securityQ.get() == "Select":
            messagebox.showerror("Error", "All fields are required",parent=self.root)
        elif self.var_pass.get() != self.var_confpass.get():
            messagebox.showerror("Error", "Password & confirm password must be the same",parent=self.root)
        elif self.var_check.get() == 0:
            messagebox.showerror("Error", "Please agree to our terms and conditions",parent=self.root)
        else:
            conn=mysql.connector.connect(host="localhost", user= "root",  password="Maaz%2006", database="maazdb")
            my_cursor=conn.cursor()
            query=("select * from register where email=%s")
            value=(self.var_email.get(),)
            my_cursor.execute(query, value)
            row=my_cursor.fetchone()
            if row!=None:
                messagebox.showerror("Error", "User already exist, plaese try another var_email",parent=self.root)
            else:
                my_cursor.execute("insert into register values (%s, %s, %s, %s, %s, %s, %s)",(
                                                           
                                                                                               self.var_fname.get(),
                                                                                               self.var_lname.get(),
                                                                                               self.var_contact.get(),
                                                                                               self.var_email.get(),
                                                                                               self.var_securityQ.get(),
                                                                                               self.var_securityA.get(),
                                                                                               self.var_pass.get(),
                                                                                               
                    
                                                                                             ))
            conn.commit()
            conn.close()
            messagebox.showinfo("Success","Register Successfully",parent=self.root)


class face_recognition_system:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        # First image
        img = Image.open(r"images/left.jpg")
        img = img.resize((1550, 800),Image.LANCZOS)
        self.photoimg = ImageTk.PhotoImage(img)

        f_lbl = Label(self.root, image=self.photoimg)
        f_lbl.place(x=0, y=0, width=1550, height=800)


        #Second image
        img1 = Image.open(r"images/center.jpg")
        img1 = img1.resize((530, 130),Image.LANCZOS)
        self.photoimg1 = ImageTk.PhotoImage(img1)

        f_lbl = Label(self.root, image=self.photoimg1)
        f_lbl.place(x=500, y=0, width=530, height=130)


        #Third image
        img2 = Image.open(r"images/right.jpg")
        img2 = img2.resize((530, 130),Image.LANCZOS)
        self.photoimg2 = ImageTk.PhotoImage(img2)

        f_lbl = Label(self.root, image=self.photoimg2)
        f_lbl.place(x=1000, y=0, width=530, height=130)


        #Background image
        img3 = Image.open(r"images/bg.jpeg")
        img3 = img3.resize((1530,710),Image.LANCZOS)
        self.photoimg3 = ImageTk.PhotoImage(img3)

        bg_img = Label(self.root, image=self.photoimg3)
        bg_img.place(x=0, y=130, width=1530, height=710)

        #Title 
        title_lbl=Label(bg_img,text="FACE RECOGNITION ATTANDANCE SYSTEM",font=("times new roman",35,"bold"),bg="lightblue",fg="black")
        title_lbl.place(x=0,y=0,width=1530,height=45)
        
        def time():
            string=strftime('%H:%M:%S %p')
            lbl.config(text = string)
            lbl.after(1000, time)
        
        lbl = Label(title_lbl,font=("times new roman",14,"bold"),background="white",foreground="blue")
        lbl.place(x=10,y=0,width=110,height=50)
        time()

        #student detail button1
        img4 = Image.open(r"images/add_user.png")
        img4 = img4.resize((350, 440),Image.LANCZOS)
        self.photoimg4 = ImageTk.PhotoImage(img4)

        b1 = Button(bg_img, image=self.photoimg4,command=self.student_details,cursor="hand2")
        b1.place(x=200, y=100, width=350, height=440)

        b1_1 = Button(bg_img, text="Student details",command=self.student_details,cursor="hand2",font=("times new roman",20,"bold"),bg="black",fg="white")
        b1_1.place(x=200, y=520, width=350, height=40)


        #detect face button
        img5 = Image.open(r"images/recog.jpg")
        img5 = img5.resize((350, 440),Image.LANCZOS)
        self.photoimg5 = ImageTk.PhotoImage(img5)

        b1 = Button(bg_img,command=self.face_data, image=self.photoimg5,cursor="hand2")
        b1.place(x=580, y=100, width=350, height=440)

        b1_1 = Button(bg_img,command=self.face_data,  text="Face detection",cursor="hand2",font=("times new roman",20,"bold"),bg="black",fg="white")
        b1_1.place(x=580, y=520, width=350, height=40)

        #attandance button
        img6 = Image.open(r"images/attendance_list.png")
        img6 = img6.resize((350, 450),Image.LANCZOS)
        self.photoimg6 = ImageTk.PhotoImage(img6)

        b1 = Button(bg_img,command=self.attandance_func, image=self.photoimg6,cursor="hand2")
        b1.place(x=960, y=100, width=350, height=450)

        b1_1 = Button(bg_img,command=self.attandance_func, text="Attandance",cursor="hand2",font=("times new roman",20,"bold"),bg="black",fg="white")
        b1_1.place(x=960, y=520, width=350, height=40)

        # #help button
        # img7 = Image.open(r"images/attendance-management.jpg")
        # img7 = img7.resize((220, 200),Image.LANCZOS)
        # self.photoimg7 = ImageTk.PhotoImage(img7)

        # b1 = Button(bg_img,command=self.help_desk, image=self.photoimg7,cursor="hand2")
        # b1.place(x=1080, y=100, width=220, height=200)

        # b1_1 = Button(bg_img,command=self.help_desk, text="Help Desk",cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
        # b1_1.place(x=1080, y=280, width=220, height=40)


        # #train face button
        # img8 = Image.open(r"images/attendance-management.jpg")
        # img8 = img8.resize((220, 200),Image.LANCZOS)
        # self.photoimg8 = ImageTk.PhotoImage(img8)

        # b1 = Button(bg_img,command=self.train_data,image=self.photoimg8,cursor="hand2")
        # b1.place(x=120, y=380, width=220, height=200)

        # b1_1 = Button(bg_img, text="Train Data",command=self.train_data ,cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
        # b1_1.place(x=120, y=560, width=220, height=40)

        # #photos button
        # img9 = Image.open(r"images/attendance-management.jpg")
        # img9 = img9.resize((220, 200),Image.LANCZOS)
        # self.photoimg9 = ImageTk.PhotoImage(img7)

        # b1 = Button(bg_img, image=self.photoimg9,cursor="hand2",command=self.open_img)
        # b1.place(x=440, y=380, width=220, height=200)

        # b1_1 = Button(bg_img, text="Stored Photos",cursor="hand2",command=self.open_img,font=("times new roman",20,"bold"),bg="darkblue",fg="white")
        # b1_1.place(x=440, y=560, width=220, height=40)

        # #developer
        # img10 = Image.open(r"images/attendance-management.jpg")
        # img10 = img10.resize((220, 200),Image.LANCZOS)
        # self.photoimg10 = ImageTk.PhotoImage(img10)

        # b1 = Button(bg_img, command=self.developer_data, image=self.photoimg10,cursor="hand2")
        # b1.place(x=760, y=380, width=220, height=200)

        # b1_1 = Button(bg_img, command=self.developer_data, text="Developer",cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
        # b1_1.place(x=760, y=560, width=220, height=40)

        #exit
        img11 = Image.open(r"images/attendance-management.jpg")
        img11 = img11.resize((90, 90),Image.LANCZOS)
        self.photoimg11 = ImageTk.PhotoImage(img11)

        # b1 = Button(bg_img, command=self.iExit, image=self.photoimg11,cursor="hand2")
        # b1.place(x=1100, y=500, width=90, height=90)

        b1_1 = Button(bg_img, command=self.iExit,text="EXIT",cursor="hand2",font=("times new roman",20,"bold"),bg="darkblue",fg="white")
        b1_1.place(x=1395, y=610, width=120, height=40)

    # opens data folder by clicking photos button
    def open_img(self):
        os.startfile("Data")
        
    def iExit(self):
        self.iExit=tkinter.messagebox.askyesno("Face Recognition","Are you Sure to Exit this project",parent=self.root)
        if self.iExit>0:
            self.root.destroy()
        else:
            return    
        
    #Function buttons

    def student_details(self):
        self.new_window=Toplevel(self.root)
        self.app=Student (self.new_window)
        
    # def train_data(self):
    #     self.new_window=Toplevel(self.root)
    #     self.app=Train(self.new_window)
        
    def face_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Face_Recognition(self.new_window)
        
    def attandance_func(self):
        self.new_window=Toplevel(self.root)
        self.app=Attandance (self.new_window)
        
    # def developer_data(self):
    #     self.new_window=Toplevel(self.root)
    #     self.app=Developer (self.new_window)
        
    # def help_desk(self):
    #     self.new_window=Toplevel(self.root)
    #     self.app=Help (self.new_window)






if __name__ == "__main__":
    main()
  