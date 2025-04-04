# from tkinter import *
# from tkinter import ttk
# from PIL import Image, ImageTk  # pip install pillow
# from tkinter import messagebox
# import mysql.connector
# import cv2  # pip install opencv-python
# import face_recognition
# import numpy as np

from tkinter.constants import *

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import face_recognition
import numpy as np
import os
import torch
import torchaudio
import sounddevice as sd
import wave
from speechbrain.inference import EncoderClassifier


class Student:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Student Management System")

        # Load the Speaker Recognition Model
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model"
        )

        self.var_dep = StringVar()
        self.var_course = StringVar()
        self.var_year = StringVar()
        self.var_semester = StringVar()
        self.var_std_id = StringVar()
        self.var_std_name = StringVar()
        self.var_div = StringVar()
        self.var_roll = StringVar()
        self.var_gender = StringVar()
        self.var_dob = StringVar()
        self.var_email = StringVar()
        self.var_phone = StringVar()
        self.var_address = StringVar()
        self.var_teacher = StringVar()

        # # First image
        # img = Image.open(r"images/left.jpg")
        # img = img.resize((520, 130), Image.LANCZOS)
        # self.photoimg = ImageTk.PhotoImage(img)

        # f_lbl = Label(self.root, image=self.photoimg)
        # f_lbl.place(x=0, y=0, width=520, height=130)

        # Second image
        # img1 = Image.open(r"images/center.jpg")
        # img1 = img1.resize((520, 130), Image.LANCZOS)
        # self.photoimg1 = ImageTk.PhotoImage(img1)

        # f_lbl = Label(self.root, image=self.photoimg1)
        # f_lbl.place(x=500, y=0, width=520, height=130)

        # # Third image
        # img2 = Image.open(r"images/right.jpg")
        # img2 = img2.resize((530, 130), Image.LANCZOS)
        # self.photoimg2 = ImageTk.PhotoImage(img2)

        # f_lbl = Label(self.root, image=self.photoimg2)
        # f_lbl.place(x=1000, y=0, width=530, height=130)

        # # Background image
        # img3 = Image.open(r"images/fill.jpg")
        # img3 = img3.resize((1530, 710), Image.LANCZOS)
        # self.photoimg3 = ImageTk.PhotoImage(img3)

        # bg_img = Label(self.root, image=self.photoimg3)
        # bg_img.place(x=0, y=130, width=1530, height=710)

        # Title
        title_lbl = Label(
            root,           
            text="STUDENT MANAGEMENT SYSTEM",
            font=("Arial", 30, "bold"),
            bg="lightgrey",
            fg="black",
            border=10,
            relief=GROOVE,
        )
        # title_lbl.place(x=0, y=0, width=1530, height=45)
        title_lbl.pack(side=TOP, fill=X)

        # frame
        main_frame = Frame(root,bd=2)
        main_frame.place(x=20, y=90, width=1320, height=630)
        # main_frame.pack(fill=BOTH, expand=1)

        # left frame
        Left_frame = LabelFrame(
            main_frame,
            bd=10,
            relief=GROOVE,
            bg="lightgrey",
            text="Enter detail",
            font=("Arial", 18, "bold"),
        )
        Left_frame.place(x=0, y=5, width=640, height=623)

        # img0 = Image.open(r"images/write.jpg")
        # img0 = img0.resize((710, 130), Image.LANCZOS)
        # self.photoimg0 = ImageTk.PhotoImage(img0)

        # f_lbl = Label(Left_frame, image=self.photoimg0)
        # f_lbl.place(x=0, y=0, width=740, height=130)

        # current course
        current_course_frame = LabelFrame(
            Left_frame,
            bd=0,
            bg="lightgrey",
            # relief="groove",
            # text="Current course",
            font=("Arial", 11, "bold"),
        )
        current_course_frame.place(x=5, y=10, width=600, height=103)

        # current course info

        # department
        dep_label = Label(
            current_course_frame,
            text="Department :",
            font=("Arial", 11, "bold"),
            bg="lightgrey",
        )
        dep_label.grid(row=0, column=0, padx=10, pady=18, sticky=W)

        dep_combo = ttk.Combobox(
            current_course_frame,
            textvariable=self.var_dep,
            font=("Arial", 11, "bold"),
            state="readonly",
        )
        dep_combo["values"] = ("Select department", "computer", "it", "civil")
        dep_combo.current(0)
        dep_combo.grid(row=0, column=1, padx=10, pady=18, sticky=W)

        # course
        dep_label = Label(
            current_course_frame,
            text="Course :",
            font=("Arial", 11, "bold"),
            bg="lightgrey",
        )
        dep_label.grid(row=0, column=2, padx=15, pady=18, sticky=W)

        course_combo = ttk.Combobox(
            current_course_frame,
            textvariable=self.var_course,
            font=("Arial", 11, "bold"),
            state="readonly",
        )
        course_combo["values"] = ("Select course", "fe", "se", "te")
        course_combo.current(0)
        course_combo.grid(row=0, column=3, padx=0, pady=18, sticky=W)

        # year
        dep_label = Label(
            current_course_frame,
            text="Year :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        dep_label.grid(row=1, column=0, padx=10, pady=9, sticky=W)

        year_combo = ttk.Combobox(
            current_course_frame,
            textvariable=self.var_year,
            font=("Arial", 11, "bold"),
            state="readonly",
        )
        year_combo["values"] = ("Select year", "2021", "2022", "2023")
        year_combo.current(0)
        year_combo.grid(row=1, column=1, padx=10, pady=9, sticky=W)

        # semester
        dep_label = Label(
            current_course_frame,
            text="Sem :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        dep_label.grid(row=1, column=2, padx=15, sticky=W)

        sem_combo = ttk.Combobox(
            current_course_frame,
            textvariable=self.var_semester,
            font=("Arial", 11, "bold"),
            state="readonly",
        )
        sem_combo["values"] = ("Select sem", "1", "2")
        sem_combo.current(0)
        sem_combo.grid(row=1, column=3, padx=0, pady=8, sticky=W)
        # sem_combo.configure(bd=5, relief="groove")

        # class student information frame
        class_student_frame = LabelFrame(
            Left_frame,
            bd=0,
            bg="lightgrey",
            relief=RIDGE,
            # text="",
            font=("Arial", 11, "bold"),
        )
        class_student_frame.place(x=5, y=120, width=602, height=460)

        # class student information

        # student id
        studentId_label = Label(
            class_student_frame,
            text="Std ID :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        # studentId_label.grid(row=0, column=0, padx=10, pady=3, sticky=W)
        studentId_label.grid(row=0, column=2, padx=0, pady=12, sticky=W)

        studentId_entry = ttk.Entry(
            class_student_frame,
            textvariable=self.var_std_id,
            width=22,
            font=("Arial", 11, "bold"),
        )
        # studentId_entry.grid(row=0, column=1, padx=10, pady=5, sticky=W)
        studentId_entry.grid(row=0, column=2, padx=80, pady=12, sticky=W)

        # student name
        studentName_label = Label(
            class_student_frame,
            text="Std Name :",
            font=("Arial", 11, "bold"),
            bg="lightgrey",
        )
        # studentName_label.grid(row=0, column=2, padx=25, pady=8, sticky=W)
        studentName_label.grid(row=0, column=0, padx=10, pady=8, sticky=W)

        studentName_entry = ttk.Entry(
            class_student_frame,
            textvariable=self.var_std_name,
            width=22,
            font=("Arial", 11, "bold"),
        )
        # studentName_entry.grid(row=0, column=3, padx=10, pady=8, sticky=W)
        studentName_entry.grid(row=0, column=1, padx=25, pady=10, sticky=W)

        # class division
        class_div_label = Label(
            class_student_frame,
            text="Division :",
            font=("Arial", 11, "bold"),
            bg="lightgrey",
        )
        class_div_label.grid(row=1, column=0, padx=10, pady=16, sticky=W)

        div_combo = ttk.Combobox(
            class_student_frame,
            textvariable=self.var_div,
            width=20,
            font=("Arial", 11, "bold"),
            state="readonly",
        )
        div_combo["values"] = ("A", "B", "C")
        div_combo.current(0)
        div_combo.grid(row=1, column=1, padx=25, pady=16, sticky=W)

        # roll no
        roll_no_label = Label(
            class_student_frame,
            text="Roll no :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        roll_no_label.grid(row=1, column=2, padx=0, pady=16, sticky=W)

        roll_no_entry = ttk.Entry(
            class_student_frame,
            textvariable=self.var_roll,
            width=22,
            font=("Arial", 11, "bold")
        )
        roll_no_entry.grid(row=1, column=2, padx=80, pady=16, sticky=W)

        # gender
        gender_label = Label(
            class_student_frame,
            text="Gender :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        gender_label.grid(row=2, column=0, padx=10, pady=12, sticky=W)

        gender_combo = ttk.Combobox(
            class_student_frame,
            textvariable=self.var_gender,
            width=20,
            font=("Arial", 11, "bold"),
            state="readonly",
        )
        gender_combo["values"] = ("Male", "Female", "Other")
        gender_combo.current(0)
        gender_combo.grid(row=2, column=1, padx=25, pady=12, sticky=W)

        # dob
        dob_label = Label(
            class_student_frame,
            text="DOB :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        dob_label.grid(row=2, column=2, padx=0, pady=12, sticky=W)

        dob_entry = ttk.Entry(
            class_student_frame,
            textvariable=self.var_dob,
            width=22,
            font=("Arial", 11, "bold"),
        )
        dob_entry.grid(row=2, column=2, padx=80, pady=12, sticky=W)

        # email
        email_label = Label(
            class_student_frame,
            text="Email :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        # email_label.grid(row=3, column=0, padx=10, pady=20, sticky=W)
        email_label.grid(row=3, column=2, padx=0, pady=20, sticky=W)

        email_entry = ttk.Entry(
            class_student_frame,
            textvariable=self.var_email,
            width=22,
            font=("Arial", 11, "bold"),
        )
        # email_entry.grid(row=3, column=1, padx=25, pady=20, sticky=W)
        email_entry.grid(row=3, column=2, padx=80, pady=20, sticky=W)

        # phone no
        phone_label = Label(
            class_student_frame,
            text="Phone no :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        # phone_label.grid(row=3, column=2, padx=0, pady=20, sticky=W)
        phone_label.grid(row=3, column=0, padx=10, pady=20, sticky=W)

        phone_entry = ttk.Entry(
            class_student_frame,
            textvariable=self.var_phone,
            width=22,
            font=("Arial", 11, "bold"),
        )
        # phone_entry.grid(row=3, column=2, padx=80, pady=20, sticky=W)
        phone_entry.grid(row=3, column=1, padx=25, pady=20, sticky=W)

        # address
        address_label = Label(
            class_student_frame,
            text="Address :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        # address_label.grid(row=4, column=0, padx=10, pady=8, sticky=W)
        address_label.grid(row=4, column=2, padx=0, pady=8, sticky=W)

        address_entry = ttk.Entry(
            class_student_frame,
            textvariable=self.var_address,
            width=22,
            font=("Arial", 11, "bold"),
        )
        # address_entry.grid(row=4, column=1, padx=25, pady=8, sticky=W)
        address_entry.grid(row=4, column=2, padx=80, pady=8, sticky=W)

        # teacher name
        teacher_label = Label(
            class_student_frame,
            text="Trs.name :",
            bg="lightgrey",
            font=("Arial", 11, "bold"),
        )
        # teacher_label.grid(row=4, column=2, padx=0,pady=8, sticky=W)
        teacher_label.grid(row=4, column=0, padx=10, pady=8, sticky=W)

        teacher_entry = ttk.Entry(
            class_student_frame,
            textvariable=self.var_teacher,
            width=22,
            font=("Arial", 11, "bold"),
        )
        # teacher_entry.grid(row=4, column=2, padx=80, pady=8, sticky=W)
        teacher_entry.grid(row=4, column=1, padx=25, pady=8, sticky=W)

        # radio buttons
        self.var_radio1 = StringVar()
        radiobtn1 = ttk.Radiobutton(
            class_student_frame,
            variable=self.var_radio1,
            text="Take photo sample",
            style="Custom.TRadiobutton",
            value="Yes",
        )
        radiobtn1.place(x=15, y=280, width=200, height=30)

        style = ttk.Style()
        style.configure(
            "Custom.TRadiobutton", font=("Arial", 12, "bold"), background="lightgrey"
        )

        radiobtn2 = ttk.Radiobutton(
            class_student_frame,
            variable=self.var_radio1,
            text="No photo sample",
            value="No",
            style="Custom.TRadiobutton",
        )
        radiobtn2.place(x=200, y=280, width=200, height=30)

        # button frame
        btn_frame = Frame(class_student_frame, bd=0, relief=RIDGE, bg="lightgrey")
        btn_frame.place(x=0, y=345, width=710, height=38)

        save_btn = Button(
            btn_frame,
            command=self.add_data,
            text="Save",
            width=15,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        save_btn.grid(row=0, column=0, padx=10, pady=10)

        update_btn = Button(
            btn_frame,
            command=self.update_data,
            text="Update",
            width=15,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        update_btn.grid(row=0, column=1, padx=12, pady=10)

        delete_btn = Button(
            btn_frame,
            text="Delete",
            command=self.delete_data,
            width=15,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        delete_btn.grid(row=0, column=2, padx=12, pady=10)

        reset_btn = Button(
            btn_frame,
            text="Reset",
            command=self.reset_data,
            width=15,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        reset_btn.grid(row=0, column=3, padx=12, pady=10)

        btn_frame1 = Frame(class_student_frame, bd=0, relief=RIDGE, bg="lightgrey")
        btn_frame1.place(x=0, y=400, width=715, height=45)

        take_photo_btn = Button(
            btn_frame1,
            command=self.generate_dataset,
            text="Take photo sample",
            width=20,
            font=("Arial", 12, "bold"),
            bg="#919294",
            fg="#000000",
        )
        take_photo_btn.grid(row=0, column=0, padx=50)

        Take_voice_sample_btn = Button(
            btn_frame1,
            command=self.record_voice_sample,
            text="Take voice sample",
            width=20,
            font=("Arial", 12, "bold"),
            bg="#919294",
            fg="#000000",
        )
        Take_voice_sample_btn.grid(row=0, column=1, padx=45)

        # right frame
        Right_frame = LabelFrame(
            main_frame,
            bd=10,
            bg="lightgrey",
            relief=GROOVE,
            text="Student details",
            font=("Arial", 18, "bold"),
        )
        Right_frame.place(x=650, y=5, width=670, height=623)

        # img99 = Image.open(r"images/spray.jpg")
        # img99 = img99.resize((690, 130), Image.LANCZOS)
        # self.photoimg99 = ImageTk.PhotoImage(img99)

        # f_lbl = Label(Right_frame, image=self.photoimg99)
        # f_lbl.place(x=10, y=0, width=690, height=130)

        # ssearch system
        search_frame = LabelFrame(
            Right_frame,
            bd=0,
            bg="lightgrey",
            relief=RIDGE,
            text="",
            font=("Arial", 11, "bold"),
        )
        search_frame.place(x=7, y=10, width=635, height=50)

        search_label = Label(
            search_frame, text="Search By:", font=("Arial", 12, "bold")
        )
        # search_label.grid(row=0, column=0, padx=0, pady=15)
        search_label.place(x=5, y=12)

        self.search_combo = ttk.Combobox(
            search_frame,
            font=("Arial", 13, "bold"),
            state="readonly",
            width=15,
        )
        self.search_combo["values"] = ("Select", "Roll no", "Phone no")
        self.search_combo.current(0)
        # self.search_combo.grid(row=0, column=1, padx=10, pady=5, sticky=W)
        self.search_combo.place(x=100, y=12, width=130)

        self.search_entry = ttk.Entry(
            search_frame, width=15, font=("Arial", 11, "bold")
        )
        # self.search_entry.grid(row=0, column=4, padx=10, pady=5, sticky=W)
        self.search_entry.place(x=235, y=12, width=190, height=25)

        search_btn = Button(
            search_frame,
            command=self.search_data,
            text="Search",
            # width=5,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        # search_btn.grid(row=0, column=3, padx=10)
        search_btn.place(x=443, y=12, width=80, height=25)

        showAll_btn = Button(
            search_frame,
            command=self.show_all_data,
            text="Show All",
            # width=8,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        # showAll_btn.grid(row=0, column=4, padx=10)
        showAll_btn.place(x=540, y=12, width=80, height=25)

        # =========================table frame=============================
        table_frame = Frame(Right_frame, bd=2, bg="lightgrey", relief=GROOVE)
        table_frame.place(x=6, y=60, width=638, height=520)

        # ============Scroll Bar==============#
        style = ttk.Style()

        # Configure default Treeview appearance
        style.configure(
            "Treeview",
            rowheight=25,
            font=("Arial", 10),
            background="white",
            fieldbackground="lightgrey",
            foreground="black",
        )

        # Configure heading (column titles)
        style.configure(
            "Treeview.Heading",
            font=("Arial", 10, "bold"),
            foreground="black",
            borderwidth=2,
            relief="ridge",
        )

        # Change selected row background to light grey & text to white
        # style.map("Treeview",
        #     background=[("selected", "lightgrey")],  # Selected row background
        #     foreground=[("selected", "black")]       # Selected row text color
        # )

        scroll_x = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame, orient=VERTICAL)

        self.student_table = ttk.Treeview(
            table_frame,
            columns=(
                "dep",
                "course",
                "year",
                "sem",
                "id",
                "name",
                "div",
                "roll",
                "gender",
                "dob",
                "email",
                "phone",
                "address",
                "teacher",
                "photo",
            ),
            xscrollcommand=scroll_x.set,
            yscrollcommand=scroll_y.set,
            style="Treeview",
        )

        self.student_table.tag_configure("oddrow", background="white")
        self.student_table.tag_configure(
            "evenrow", background="#b0b0b0", foreground="black"
        )

        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)

        self.student_table.heading("dep", text="Department")
        self.student_table.heading("course", text="Course")
        self.student_table.heading("year", text="Year")
        self.student_table.heading("sem", text="Semester")
        self.student_table.heading("id", text="ID")
        self.student_table.heading("name", text="Name")
        self.student_table.heading("div", text="Division")
        self.student_table.heading("roll", text="Roll No")
        self.student_table.heading("gender", text="Gender")
        self.student_table.heading("dob", text="D.O.B")
        self.student_table.heading("email", text="Email")
        self.student_table.heading("phone", text="Phone")
        self.student_table.heading("address", text="Address")
        self.student_table.heading("teacher", text="Teacher")
        self.student_table.heading("photo", text="Photo")
        self.student_table["show"] = "headings"

        self.student_table.column("dep", width=100)
        self.student_table.column("course", width=100)
        self.student_table.column("year", width=100)
        self.student_table.column("sem", width=100)
        self.student_table.column("id", width=100)
        self.student_table.column("name", width=100)
        self.student_table.column("div", width=100)
        self.student_table.column("roll", width=100)
        self.student_table.column("gender", width=100)
        self.student_table.column("dob", width=100)
        self.student_table.column("email", width=120)
        self.student_table.column("phone", width=100)
        self.student_table.column("address", width=100)
        self.student_table.column("teacher", width=100)
        self.student_table.column("photo", width=150)

        self.student_table.pack(fill=BOTH, expand=1)

        self.student_table.bind("<ButtonRelease>", self.get_cursor)

        self.fetch_data()

        # ================== Function decration ================

    def add_data(self):

        if (
            self.var_dep.get() == "Select department"
            or self.var_course.get() == "Select course"
            or self.var_year.get() == "Select year"
            or self.var_semester.get() == "Select sem"
            or self.var_std_id.get() == ""
            or self.var_std_name.get() == ""
            or self.var_div.get() == ""
            or self.var_roll.get() == ""
            or self.var_gender.get() == ""
            or self.var_dob.get() == ""
            or self.var_email.get() == ""
            or self.var_phone.get() == ""
            or self.var_address.get() == ""
            or self.var_teacher.get() == ""
            or self.var_radio1.get() == ""
        ):

            messagebox.showerror("Error", "All fields are required", parent=self.root)

        else:

            try:
                conn = mysql.connector.connect(
                    host="localhost",
                    username="root",
                    password="Maaz%2006",
                    database="maazdb",
                )
                my_cursor = conn.cursor()
                my_cursor.execute(
                    "insert into students values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        self.var_dep.get(),
                        self.var_course.get(),
                        self.var_year.get(),
                        self.var_semester.get(),
                        self.var_std_id.get(),
                        self.var_std_name.get(),
                        self.var_div.get(),
                        self.var_roll.get(),
                        self.var_gender.get(),
                        self.var_dob.get(),
                        self.var_email.get(),
                        self.var_phone.get(),
                        self.var_address.get(),
                        self.var_teacher.get(),
                        self.var_radio1.get(),
                    ),
                )
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo(
                    "Success",
                    "Student details has been added successfully",
                    parent=self.root,
                )

            except Exception as es:
                messagebox.showerror("Error", f"Due to : {str(es)}", parent=self.root)

    # def fetch_data(self):
    #     conn = mysql.connector.connect(
    #         host="localhost", username="root", password="Maaz%2006", database="maazdb"
    #     )
    #     my_cursor = conn.cursor()
    #     my_cursor.execute("Select * from students")
    #     data = my_cursor.fetchall()

    #     if len(data) != 0:
    #         self.student_table.delete(*self.student_table.get_children())
    #         for i in data:
    #             self.student_table.insert("", END, values=i)
    #         conn.commit()

    #     conn.close()

    def fetch_data(self):
        conn = mysql.connector.connect(
            host="localhost", username="root", password="Maaz%2006", database="maazdb"
        )
        my_cursor = conn.cursor()
        my_cursor.execute("Select * from students")
        data = my_cursor.fetchall()

        if len(data) != 0:
            self.student_table.delete(*self.student_table.get_children())
            for i, row in enumerate(data):
                if i % 2 == 0:  # Even index (0, 2, 4...) - lightgrey
                    self.student_table.insert("", END, values=row, tags=("evenrow",))
                else:  # Odd index (1, 3, 5...) - white
                    self.student_table.insert("", END, values=row, tags=("oddrow",))
            conn.commit()

        conn.close()

        # ============== get cursor ==================

    def get_cursor(self, event=""):
        cursor_focus = self.student_table.focus()
        content = self.student_table.item(cursor_focus)
        data = content["values"]

        self.var_dep.set(data[0]),
        self.var_course.set(data[1]),
        self.var_year.set(data[2]),
        self.var_semester.set(data[3]),
        self.var_std_id.set(data[4]),
        self.var_std_name.set(data[5]),
        self.var_div.set(data[6]),
        self.var_roll.set(data[7]),
        self.var_gender.set(data[8]),
        self.var_dob.set(data[9]),
        self.var_email.set(data[10]),
        self.var_phone.set(data[11]),
        self.var_address.set(data[12]),
        self.var_teacher.set(data[13]),
        self.var_radio1.set(data[14])

    # ======================== update function ==========================
    def update_data(self):

        if (
            self.var_dep.get() == "Select department"
            or self.var_course.get() == "Select course"
            or self.var_year.get() == "Select year"
            or self.var_semester.get() == "Select sem"
            or self.var_std_id.get() == ""
            or self.var_std_name.get() == ""
            or self.var_div.get() == ""
            or self.var_roll.get() == ""
            or self.var_gender.get() == ""
            or self.var_dob.get() == ""
            or self.var_email.get() == ""
            or self.var_phone.get() == ""
            or self.var_address.get() == ""
            or self.var_teacher.get() == ""
            or self.var_radio1.get() == ""
        ):

            messagebox.showerror("Error", "All fields are required", parent=self.root)

        else:
            try:
                Update = messagebox.askyesno(
                    "Update",
                    "Do you want to update this student details",
                    parent=self.root,
                )
                if Update > 0:
                    conn = mysql.connector.connect(
                        host="localhost",
                        username="root",
                        password="Maaz%2006",
                        database="maazdb",
                    )
                    my_cursor = conn.cursor()
                    my_cursor.execute(
                        "update students set Dep=%s,Course=%s,Year=%s,Semester=%s,Name=%s,Division=%s,Roll=%s,Gender=%s,Dob=%s,Email=%s,Phone=%s,Address=%s,Teacher=%s,PhotoSample=%s where Student_id=%s",
                        (
                            self.var_dep.get(),
                            self.var_course.get(),
                            self.var_year.get(),
                            self.var_semester.get(),
                            self.var_std_name.get(),
                            self.var_div.get(),
                            self.var_roll.get(),
                            self.var_gender.get(),
                            self.var_dob.get(),
                            self.var_email.get(),
                            self.var_phone.get(),
                            self.var_address.get(),
                            self.var_teacher.get(),
                            self.var_radio1.get(),
                            self.var_std_id.get(),
                        ),
                    )
                else:
                    if not Update:
                        return

                messagebox.showinfo(
                    "Success", "Student deatail updated successfully", parent=self.root
                )
                conn.commit()
                self.fetch_data()
                conn.close()

            except Exception as es:
                messagebox.showerror("Error", f"Due to {str(es)}", parent=self.root)

    # ========================= delete data =============================
    def delete_data(self):

        if self.var_std_id.get() == "":
            messagebox.showerror(
                "Error", "Student id must be required ", parent=self.root
            )

        else:
            try:
                delete = messagebox.askyesno(
                    "Student Delete page",
                    "Do you want to delete this student detail",
                    parent=self.root,
                )
                if delete > 0:
                    conn = mysql.connector.connect(
                        host="localhost",
                        username="root",
                        password="Maaz%2006",
                        database="maazdb",
                    )
                    my_cursor = conn.cursor()
                    my_cursor.execute(
                        "delete from students where Student_id=%s",
                        (self.var_std_id.get(),),
                    )
                else:
                    if not delete:
                        return

                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo(
                    "Delete", "Student deatail deleted successfully", parent=self.root
                )

            except Exception as es:
                messagebox.showerror("Error", f"Due to {str(es)}", parent=self.root)

    def reset_data(self):
        self.var_dep.set("Select Department"),
        self.var_course.set("Select Course"),
        self.var_year.set("Select Year"),
        self.var_semester.set("Select Semester"),
        self.var_std_id.set("")
        self.var_std_name.set(""),
        self.var_div.set(""),
        self.var_roll.set(""),
        self.var_gender.set(""),
        self.var_dob.set(""),
        self.var_email.set(""),
        self.var_phone.set(""),
        self.var_address.set(""),
        self.var_teacher.set(""),
        self.var_radio1.set("")

    # def search_data(self):
    #     search_by = self.search_combo.get()  # Get the selected search filter
    #     search_value = self.search_entry.get()  # Get the entered search term

    #     if search_by == "Select" or search_value == "":
    #         messagebox.showerror(
    #             "Error",
    #             "Please select a search criterion and enter a value.",
    #             parent=self.root,
    #         )
    #         return

    #     # Map UI selection to actual database column names
    #     column_mapping = {"Roll no": "Roll", "Phone no": "Phone"}
    #     column_name = column_mapping.get(search_by, None)

    #     if not column_name:
    #         messagebox.showerror(
    #             "Error", "Invalid search criterion selected.", parent=self.root
    #         )
    #         return

    #     try:
    #         conn = mysql.connector.connect(
    #             host="localhost",
    #             username="root",
    #             password="Maaz%2006",
    #             database="maazdb",
    #         )
    #         my_cursor = conn.cursor()
    #         query = f"SELECT * FROM students WHERE {column_name} LIKE %s"
    #         my_cursor.execute(query, ("%" + search_value + "%",))
    #         rows = my_cursor.fetchall()

    #         if len(rows) != 0:
    #             self.student_table.delete(
    #                 *self.student_table.get_children()
    #             )  # Clear current table data
    #             for row in rows:
    #                 self.student_table.insert("", "end", values=row)
    #         else:
    #             messagebox.showinfo("No Results", "No matching records found.")

    #         conn.close()
    #     except Exception as es:
    #         messagebox.showerror(
    #             "Error", f"Database error: {str(es)}", parent=self.root
    #         )

    def search_data(self):
        search_by = self.search_combo.get()
        search_value = self.search_entry.get()

        if search_by == "Select" or search_value == "":
            messagebox.showerror(
                "Error",
                "Please select a search criterion and enter a value.",
                parent=self.root,
            )
            return

        column_mapping = {"Roll no": "Roll", "Phone no": "Phone"}
        column_name = column_mapping.get(search_by, None)

        if not column_name:
            messagebox.showerror(
                "Error", "Invalid search criterion selected.", parent=self.root
            )
            return

        try:
            conn = mysql.connector.connect(
                host="localhost",
                username="root",
                password="Maaz%2006",
                database="maazdb",
            )
            my_cursor = conn.cursor()
            query = f"SELECT * FROM students WHERE {column_name} LIKE %s"
            my_cursor.execute(query, ("%" + search_value + "%",))
            rows = my_cursor.fetchall()

            if len(rows) != 0:
                self.student_table.delete(*self.student_table.get_children())
                for i, row in enumerate(rows):  # Added enumerate for row index
                    if i % 2 == 0:  # Even rows (0, 2, 4...) - lightgrey
                        self.student_table.insert(
                            "", END, values=row, tags=("evenrow",)
                        )
                    else:  # Odd rows (1, 3, 5...) - white
                        self.student_table.insert("", END, values=row, tags=("oddrow",))
            else:
                messagebox.showinfo("No Results", "No matching records found.")

            conn.close()
        except Exception as es:
            messagebox.showerror(
                "Error", f"Database error: {str(es)}", parent=self.root
            )

    def show_all_data(self):
        try:
            conn = mysql.connector.connect(
                host="localhost",
                username="root",
                password="Maaz%2006",
                database="maazdb",
            )
            my_cursor = conn.cursor()
            my_cursor.execute("SELECT * FROM students")
            rows = my_cursor.fetchall()

            if len(rows) != 0:
                self.student_table.delete(
                    *self.student_table.get_children()
                )  # Clear current table data
                for row in rows:
                    self.student_table.insert("", "end", values=row)
            else:
                messagebox.showinfo("No Data", "No records found in the database.")

            conn.close()
        except Exception as es:
            messagebox.showerror(
                "Error", f"Database error: {str(es)}", parent=self.root
            )

    # =================generating dataset =====================

    def generate_dataset(self):
        if (
            self.var_dep.get() == "Select department"
            or self.var_course.get() == "Select course"
            or self.var_year.get() == "Select year"
            or self.var_semester.get() == "Select sem"
            or self.var_std_id.get() == ""
            or self.var_std_name.get() == ""
            or self.var_div.get() == ""
            or self.var_roll.get() == ""
            or self.var_gender.get() == ""
            or self.var_dob.get() == ""
            or self.var_email.get() == ""
            or self.var_phone.get() == ""
            or self.var_address.get() == ""
            or self.var_teacher.get() == ""
            or self.var_radio1.get() == ""
        ):

            messagebox.showerror("Error", "All fields are required", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(
                    host="localhost",
                    username="root",
                    password="Maaz%2006",
                    database="maazdb",
                )
                my_cursor = conn.cursor()

                # Ensure student details are updated in the database
                my_cursor.execute(
                    """
                    UPDATE students SET 
                    Dep=%s, Course=%s, Year=%s, Semester=%s, Name=%s, Division=%s, Roll=%s, Gender=%s, 
                    Dob=%s, Email=%s, Phone=%s, Address=%s, Teacher=%s, PhotoSample=%s 
                    WHERE Student_id=%s
                """,
                    (
                        self.var_dep.get(),
                        self.var_course.get(),
                        self.var_year.get(),
                        self.var_semester.get(),
                        self.var_std_name.get(),
                        self.var_div.get(),
                        self.var_roll.get(),
                        self.var_gender.get(),
                        self.var_dob.get(),
                        self.var_email.get(),
                        self.var_phone.get(),
                        self.var_address.get(),
                        self.var_teacher.get(),
                        self.var_radio1.get(),
                        self.var_std_id.get(),
                    ),
                )

                conn.commit()
                conn.close()

                # ================= Face Capture =================
                cap = cv2.VideoCapture(0)  # Start webcam
                img_id = 0  # Ensure new count for every student
                student_id = str(
                    self.var_std_id.get()
                )  # Convert student ID to string for filenames

                while img_id < 100:
                    ret, my_frame = cap.read()
                    if not ret:
                        continue  # Skip if frame not captured

                    # Convert the frame to RGB (required by face_recognition)
                    rgb_frame = cv2.cvtColor(my_frame, cv2.COLOR_BGR2RGB)

                    # Detect faces in the frame
                    face_locations = face_recognition.face_locations(rgb_frame)
                    if len(face_locations) == 0:
                        continue  # Skip if no face is detected

                    # Generate face encodings
                    face_encodings = face_recognition.face_encodings(
                        rgb_frame, face_locations
                    )

                    for face_encoding in face_encodings:
                        img_id += 1
                        # Save the face encoding to a file
                        file_name_path = f"data/user.{student_id}.{img_id}.npy"
                        np.save(file_name_path, face_encoding)

                        # Display the face in a window
                        face = cv2.resize(my_frame, (450, 450))
                        cv2.putText(
                            face,
                            f"ID {student_id} Img {img_id}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow("Cropped Face", face)

                        if cv2.waitKey(1) == 13:  # If 'Enter' key is pressed, break
                            break

                cap.release()
                cv2.destroyAllWindows()

                messagebox.showinfo(
                    "Success",
                    f"Dataset generated for student ID: {student_id}",
                    Parent=self.root,
                )

            except Exception as es:
                messagebox.showerror("Error", f"Due to {str(es)}", Parent=self.root)
            finally:
                cap.release()
                cv2.destroyAllWindows()

    def record_voice_sample(self):
        student_id = self.var_std_id.get()

        if not student_id:
            messagebox.showerror("Error", "Please enter a valid Student ID first.")
            return

        # Fetch the student's name from the database
        try:
            conn = mysql.connector.connect(
                host="localhost",
                username="root",
                password="Maaz%2006",
                database="maazdb",
            )
            my_cursor = conn.cursor()
            my_cursor.execute(
                "SELECT Name FROM students WHERE Student_id=%s", (student_id,)
            )
            result = my_cursor.fetchone()

            if not result:
                messagebox.showerror(
                    "Error",
                    "Student ID not found in database. Please enter a valid ID.",
                )
                conn.close()
                return

            student_name = result[0]  # Correct student name from the database
            conn.close()

        except Exception as e:
            messagebox.showerror("Database Error", f"Error fetching student name: {e}")
            return

        fs = 16000  # Sample rate
        duration = 5  # Duration of recording

        messagebox.showinfo(
            "Recording", f"Recording for {student_name} will start. Speak clearly."
        )
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
        sd.wait()

        voice_folder = "voiceprints"
        os.makedirs(voice_folder, exist_ok=True)
        filename = os.path.join(voice_folder, f"{student_id}_{student_name}.wav")

        wavefile = wave.open(filename, "wb")
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(fs)
        wavefile.writeframes(recording.tobytes())
        wavefile.close()

        # Extract voiceprint
        signal, fs = torchaudio.load(filename)
        voiceprint = self.classifier.encode_batch(signal)
        voiceprint = voiceprint.squeeze(0).squeeze(0).detach().numpy()

        npy_filename = filename.replace(".wav", ".npy")
        np.save(npy_filename, voiceprint)

        messagebox.showinfo("Success", f"Voice sample saved as {filename}")


if __name__ == "__main__":
    root = Tk()
    obj = Student(root)
    root.mainloop()
