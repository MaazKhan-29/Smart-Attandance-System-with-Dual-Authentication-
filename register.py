# from tkinter import *
# from tkinter import ttk
# from PIL import Image, ImageTk  # pip install pillow
# from tkinter import messagebox
# import mysql.connector

# class Register:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Register")
#         self.root.geometry("1600x900+0+0")
        
#         #=============== Variablees =====================
#         self.var_fname=StringVar()
#         self.var_lname=StringVar()
#         self.var_contact=StringVar()
#         self.var_email=StringVar()
#         self.var_securityQ=StringVar()
#         self.var_securityA=StringVar()
#         self.var_pass=StringVar()
#         self.var_confpass=StringVar()
        
#         img00 = Image.open(r"images/attendance_management.jpeg")
#         img00 = img00.resize((1550, 800),  Image.LANCZOS)
#         self.photoimage00 = ImageTk.PhotoImage(img00)
#         lblimg0 = Label(self.root, image=self.photoimage00, borderwidth=0)
#         lblimg0.place(x=0, y=0, width=1550, height=800)
        
#         img99 = Image.open(r"images/attendance_management.jpeg")
#         img99 = img99.resize((1550, 800),  Image.LANCZOS)
#         self.photoimage99 = ImageTk.PhotoImage(img99)
#         lblimg0 = Label(self.root, image=self.photoimage99, borderwidth=0)
#         lblimg0.place(x=230, y=100, width=300, height=550)
        
#         # Main Frame
#         frame = Frame(self.root, bg="white")
#         frame.place(x=520, y=100, width=800, height=550)
        
#         register_lbl = Label(frame, text="REGISTER HERE", font=("times new roman", 20, "bold"), fg="darkgreen", bg="white")
#         register_lbl.place(x=20, y=20)
        
#         # Labels and Entry Fields
#         # Row 1
#         fname = Label(frame, text="First Name", font=("times new roman", 15, "bold"), bg="white", fg="black")
#         fname.place(x=50, y=100)
#         self.fname_entry = ttk.Entry(frame,textvariable=self.var_fname, font=("times new roman", 15, "bold"))
#         self.fname_entry.place(x=50, y=130, width=250)
        
#         l_name = Label(frame, text="Last Name", font=("times new roman", 15, "bold"), bg="white", fg="black")
#         l_name.place(x=370, y=100)
#         self.txt_lname = ttk.Entry(frame,textvariable=self.var_lname, font=("times new roman", 15))
#         self.txt_lname.place(x=370, y=130, width=250)
        
#         # Row 2
#         contact = Label(frame, text="Contact No", font=("times new roman", 15, "bold"), bg="white", fg="black")
#         contact.place(x=50, y=170)
#         self.txt_contact = ttk.Entry(frame,textvariable=self.var_contact, font=("times new roman", 15))
#         self.txt_contact.place(x=50, y=200, width=250)
        
#         email = Label(frame, text="Email", font=("times new roman", 15, "bold"), bg="white", fg="black")
#         email.place(x=370, y=170)
#         self.txt_email = ttk.Entry(frame,textvariable=self.var_email, font=("times new roman", 15))
#         self.txt_email.place(x=370, y=200, width=250)
        
#         # Row 3
#         security_Q = Label(frame, text="Select Security Question", font=("times new roman", 15, "bold"), bg="white", fg="black")
#         security_Q.place(x=50, y=240)
        
#         self.combo_security_Q = ttk.Combobox(frame,textvariable=self.var_securityQ, font=("times new roman", 15, "bold"), state="readonly")
#         self.combo_security_Q["values"] = ("Select", "Your Birth Place", "Your Girlfriend Name", "Your Pet Name")
#         self.combo_security_Q.place(x=50, y=270, width=250)
#         self.combo_security_Q.current(0)
        
#         security_A = Label(frame, text="Security Answer", font=("times new roman", 15, "bold"), bg="white", fg="black")
#         security_A.place(x=370, y=240)
#         self.txt_security = ttk.Entry(frame,textvariable=self.var_securityA, font=("times new roman", 15))
#         self.txt_security.place(x=370, y=270, width=250)
        
#         # Row 4
#         pswd = Label(frame, text="Password", font=("times new roman", 15, "bold"), bg="white", fg="black")
#         pswd.place(x=50, y=310)
#         self.txt_pswd = ttk.Entry(frame,textvariable=self.var_pass,font=("times new roman", 15))
#         self.txt_pswd.place(x=50, y=340, width=250)
        
#         confirm_pswd = Label(frame, text="Confirm Password", font=("times new roman", 15, "bold"), bg="white", fg="black")
#         confirm_pswd.place(x=370, y=310)
#         self.txt_confirm_pswd = ttk.Entry(frame,textvariable=self.var_confpass, font=("times new roman", 15))
#         self.txt_confirm_pswd.place(x=370, y=340, width=250)
        
#         # Checkbox
#         self.var_check=IntVar()
#         checkbtn = Checkbutton(frame,variable=self.var_check, text="I Agree The Terms & Conditions", font=("times new roman", 12, "bold"), onvalue=1, offvalue=0)
#         checkbtn.place(x=50, y=380)
        
#         # Buttons
#         img = Image.open(r"images/attendance-management.jpg")
#         img = img.resize((200, 55),  Image.LANCZOS)
#         self.photoimage = ImageTk.PhotoImage(img)
        
#         b1 = Button(frame, command=self.register_data, image=self.photoimage, borderwidth=0, cursor="hand2", font=("times new roman", 15, "bold"), fg="white")
#         b1.place(x=100, y=450, width=200)
        
#         img1 = Image.open(r"images/attendance-management.jpg")
#         img1 = img1.resize((200, 55),  Image.LANCZOS)
#         self.photoimage1 = ImageTk.PhotoImage(img1)
        
#         b1 = Button(frame, image=self.photoimage1, borderwidth=0, cursor="hand2", font=("times new roman", 15, "bold"), fg="white")
#         b1.place(x=360, y=450, width=200)
        
#     # =================Function declaration============================
  
#     def register_data(self):
#         if self.var_fname.get() == "" or self.var_email.get() == "" or self.var_securityQ.get() == "Select":
#             messagebox.showerror("Error", "All fields are required")
#         elif self.var_pass.get() != self.var_confpass.get():
#             messagebox.showerror("Error", "Password & confirm password must be the same")
#         elif self.var_check.get() == 0:
#             messagebox.showerror("Error", "Please agree to our terms and conditions")
#         else:
#             conn=mysql.connector.connect(host="localhost", user= "root", password="#include stdio.h687", database="face_recognizer")
#             my_cursor=conn.cursor()
#             query=("select * from register where email=%s")
#             value=(self.var_email.get(),)
#             my_cursor.execute (query, value)
#             row=my_cursor.fetchone()
#             if row!=None:
#                 messagebox.showerror("Error", "User already exist, plaese try another email")
#             else:
#                 my_cursor.execute("insert into register values (%s, %s, %s, %s, %s, %s, %s)",(
                                                           
#                                                                                                self.var_fname.get(),
#                                                                                                self.var_lname.get(),
#                                                                                                self.var_contact.get(),
#                                                                                                self.var_email.get(),
#                                                                                                self.var_securityQ.get(),
#                                                                                                self.var_securityA.get(),
#                                                                                                self.var_pass.get(),
                                                                                               
                    
#                                                                                              ))
#             conn.commit()
#             conn.close()
#             messagebox.showinfo("Success","Register Successfully")




# if __name__ == "__main__":
#     root = Tk()
#     app = Register(root)
#     root.mainloop()
