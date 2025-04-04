from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk  # pip install pillow
from tkinter import messagebox
import mysql.connector  # pip install mysql-connector-python
import cv2  # pip install opencv-python
import os
import numpy as np
import csv
from tkinter import filedialog

mydata = []


class Attandance:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Student Attendance System")

        # ================= Variables ======================
        self.var_atten_id = StringVar()
        self.var_atten_roll = StringVar()
        self.var_atten_name = StringVar()
        self.var_atten_dep = StringVar()
        self.var_atten_time = StringVar()
        self.var_atten_date = StringVar()
        self.var_atten_attandance = StringVar()

        # # First image
        # img = Image.open(r"images/eye2.jpg")
        # img = img.resize((800, 200), Image.LANCZOS)
        # self.photoimg = ImageTk.PhotoImage(img)

        # f_lbl = Label(self.root, image=self.photoimg)
        # f_lbl.place(x=0, y=0, width=800, height=200)

        # # Second image
        # img1 = Image.open(r"images/finger.jpg")
        # img1 = img1.resize((800, 200), Image.LANCZOS)
        # self.photoimg1 = ImageTk.PhotoImage(img1)

        # f_lbl = Label(self.root, image=self.photoimg1)
        # f_lbl.place(x=800, y=0, width=800, height=200)

        #  # Background image
        # img3 = Image.open(r"images/attendance-management.jpg")
        # img3 = img3.resize((1530, 710), Image.LANCZOS)
        # self.photoimg3 = ImageTk.PhotoImage(img3)

        # bg_img = Label(self.root, image=self.photoimg3)
        # bg_img.place(x=0, y=200, width=1530, height=710)

        # Title
        title_lbl = Label(
            self.root,
            text="STUDENT ATTANDANCE SYSTEM",
            font=("times new roman", 30, "bold"),
            bg="lightgrey",
            fg="black",
            border=10,
            relief=GROOVE,
        )
        title_lbl.pack(side=TOP, fill=X)
        # frame
        main_frame = Frame(self.root, bd=2)
        main_frame.place(x=20, y=90, width=1320, height=630)

        # left frame
        Left_frame = LabelFrame(
            main_frame,
            bd=10,
            relief=GROOVE,
            bg="lightgrey",
            text="Student Information",
            font=("Arial", 18, "bold"),
        )
        Left_frame.place(x=15, y=8, width=500, height=623)

        # img0 = Image.open(r"images/attendance-management.jpg")
        # img0 = img0.resize((710, 130), Image.LANCZOS)
        # self.photoimg0 = ImageTk.PhotoImage(img0)

        # f_lbl = Label(Left_frame, image=self.photoimg0)
        # f_lbl.place(x=5, y=0, width=720, height=130)

        # left inside frame
        left_inside_frame = LabelFrame(
            Left_frame,
            bd=0,
            bg="lightgrey",
            relief=RIDGE,
            font=("times new roman", 11, "bold"),
        )
        left_inside_frame.place(x=10, y=5, width=457, height=570)

        # labels entry
        # Attandance id
        studentId_label = Label(
            left_inside_frame,
            text="Attandance ID :",
            font=("times new roman", 14, "bold"),
        )
        studentId_label.grid(row=0, column=0, padx=10, pady=20, sticky=W)

        studentId_entry = ttk.Entry(
            left_inside_frame,
            textvariable=self.var_atten_id,
            width=20,
            font=("times new roman", 14, "bold"),
        )
        studentId_entry.grid(row=0, column=1, padx=10, pady=25, sticky=W)

        # student Roll No
        roll_no_label = Label(
            left_inside_frame, text="Roll no :", font=("times new roman", 14, "bold")
        )
        roll_no_label.grid(row=1, column=0, padx=10, pady=8, sticky=W)

        roll_no_entry = ttk.Entry(
            left_inside_frame,
            textvariable=self.var_atten_roll,
            width=20,
            font=("times new roman", 14, "bold"),
        )
        roll_no_entry.grid(row=1, column=1, padx=10, pady=8, sticky=W)

        # name
        studentName_label = Label(
            left_inside_frame, text="Name :", font=("times new roman", 14, "bold")
        )
        studentName_label.grid(row=2, column=0, padx=10, pady=20, sticky=W)

        studentName_entry = ttk.Entry(
            left_inside_frame,
            textvariable=self.var_atten_name,
            width=20,
            font=("times new roman", 14, "bold"),
        )
        studentName_entry.grid(row=2, column=1, padx=10, pady=25, sticky=W)

        # Department
        Department_label = Label(
            left_inside_frame, text="Department :", font=("times new roman", 14, "bold")
        )
        Department_label.grid(row=3, column=0, padx=10, pady=8, sticky=W)

        Department_entry = ttk.Entry(
            left_inside_frame,
            textvariable=self.var_atten_dep,
            width=20,
            font=("times new roman", 14, "bold"),
        )
        Department_entry.grid(row=3, column=1, padx=10, pady=8, sticky=W)

        # Time
        Time_label = Label(
            left_inside_frame, text="Time :", font=("times new roman", 14, "bold")
        )
        Time_label.grid(row=4, column=0, padx=10, pady=25, sticky=W)

        Time_entry = ttk.Entry(
            left_inside_frame,
            textvariable=self.var_atten_time,
            width=20,
            font=("times new roman", 14, "bold"),
        )
        Time_entry.grid(row=4, column=1, padx=10, pady=25, sticky=W)

        # Date
        Date_label = Label(
            left_inside_frame, text="Date :", font=("times new roman", 14, "bold")
        )
        Date_label.grid(row=5, column=0, padx=10, pady=8, sticky=W)

        Date_entry = ttk.Entry(
            left_inside_frame,
            textvariable=self.var_atten_date,
            width=20,
            font=("times new roman", 14, "bold"),
        )
        Date_entry.grid(row=5, column=1, padx=10, pady=8, sticky=W)

        # Attendance Status
        Att_status_label = Label(
            left_inside_frame,
            text="Attandance Status :",
            font=("times new roman", 14, "bold"),
        )
        Att_status_label.grid(row=6, column=0, padx=10, pady=25, sticky=W)

        Att_status_entry = ttk.Combobox(
            left_inside_frame,
            textvariable=self.var_atten_attandance,
            width=18,
            font=("times new roman", 14, "bold"),
            state="readonly",
        )
        Att_status_entry["values"] = ("Status", "Present", "Absent")
        Att_status_entry.current(0)
        Att_status_entry.grid(row=6, column=1, padx=10, pady=25, sticky=W)

        # button frame
        btn_frame = Frame(left_inside_frame, bd=0, relief=RIDGE, bg="lightgrey")
        btn_frame.place(x=0, y=460, width=457, height=80)

        save_btn = Button(
            btn_frame,
            command=self.importCsv,
            text="Import CSV",
            width=15,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        save_btn.grid(row=0, column=0,padx=50)

        Export_btn = Button(
            btn_frame,
            command=self.exportCsv,
            text="EXport CSV",
            width=15,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        Export_btn.grid(row=0, column=1,padx=45)

        delete_btn = Button(
            btn_frame,
            text="Delete",
            width=15,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        delete_btn.grid(row=1, column=0,pady=15)

        reset_btn = Button(
            btn_frame,
            command=self.reset_data,
            text="Reset",
            width=15,
            font=("Arial", 10, "bold"),
            bg="#919294",
            fg="#000000",
        )
        reset_btn.grid(row=1, column=1,pady=15)

        # right frame
        Right_frame = LabelFrame(
            main_frame,
            bd=10,
            bg="lightgrey",
            relief=GROOVE,
            text="Student details",
           font=("Arial", 18, "bold"),
        )
        Right_frame.place(x=530, y=8, width=770, height=623)

        # table frame
        table_frame = LabelFrame(
            Right_frame,
            bd=2,
            bg="white",
            relief=RIDGE,
            font=("times new roman", 11, "bold"),
        )
        table_frame.place(x=10, y=5, width=730, height=560)

        # ============Scroll Bar==============#
        scroll_x = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame, orient=VERTICAL)

        self.AttandanceReportTable = ttk.Treeview(
            table_frame,
            columns=("id", "roll", "name", "department", "time", "date", "attandance"),
            xscrollcommand=scroll_x.set,
            yscrollcommand=scroll_y.set,
        )

        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.AttandanceReportTable.xview)
        scroll_y.config(command=self.AttandanceReportTable.yview)

        self.AttandanceReportTable.heading("id", text=" Attandance ID")
        self.AttandanceReportTable.heading("roll", text="Roll ")
        self.AttandanceReportTable.heading("name", text="Name")
        self.AttandanceReportTable.heading("department", text="Department")
        self.AttandanceReportTable.heading("time", text="Time")
        self.AttandanceReportTable.heading("date", text="Date")
        self.AttandanceReportTable.heading("attandance", text="Status")

        self.AttandanceReportTable["show"] = "headings"

        self.AttandanceReportTable.column("id", width=100)
        self.AttandanceReportTable.column("roll", width=100)
        self.AttandanceReportTable.column("name", width=100)
        self.AttandanceReportTable.column("department", width=100)
        self.AttandanceReportTable.column("time", width=100)
        self.AttandanceReportTable.column("date", width=100)
        self.AttandanceReportTable.column("attandance", width=100)

        self.AttandanceReportTable.pack(fill=BOTH, expand=1)

        self.AttandanceReportTable.bind("<ButtonRelease>", self.get_cursor)

    def fetchData(self, rows):

        self.AttandanceReportTable.delete(*self.AttandanceReportTable.get_children())

        for i in rows:
            self.AttandanceReportTable.insert("", END, values=i)

    def importCsv(self):

        global mydata
        mydata.clear()

        fln = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Open CSV",
            filetypes=(("CSV File", "*csv"), ("All file", "*.*")),
            parent=self.root,
        )

        with open(fln) as myfile:
            csvread = csv.reader(myfile, delimiter=",")

            for i in csvread:
                mydata.append(i)

        self.fetchData(mydata)

    def exportCsv(self):

        try:
            if len(mydata) < 1:
                messagebox.showerror(
                    "No Data", "No Data found to export", parent=self.root
                )
                return False

            fln = filedialog.asksaveasfilename(
                initialdir=os.getcwd(),
                title="Save CSV",
                filetypes=(("CSV File", "*.csv"), ("All file", "*.*")),
                parent=self.root,
            )

            with open(fln, mode="w", newline="") as myfile:
                exp_write = csv.writer(myfile, delimiter=",")

                for i in mydata:
                    exp_write.writerow(i)

            messagebox.showinfo(
                "Data Export",
                "Your data exported to " + os.path.basename(fln) + " successfully",
            )

        except Exception as es:
            messagebox.showerror("Error", f"Due To : {str(es)}", parent=self.root)

    def get_cursor(self, event=""):

        cursor_row = self.AttandanceReportTable.focus()
        content = self.AttandanceReportTable.item(cursor_row)
        rows = content["values"]

        self.var_atten_id.set(rows[0])
        self.var_atten_roll.set(rows[1])
        self.var_atten_name.set(rows[2])
        self.var_atten_dep.set(rows[3])
        self.var_atten_time.set(rows[4])
        self.var_atten_date.set(rows[5])
        self.var_atten_attandance.set(rows[6])

    def reset_data(self, event=""):

        self.var_atten_id.set("")
        self.var_atten_roll.set("")
        self.var_atten_name.set("")
        self.var_atten_dep.set("")
        self.var_atten_time.set("")
        self.var_atten_date.set("")
        self.var_atten_attandance.set("")


if __name__ == "__main__":
    root = Tk()
    obj = Attandance(root)
    root.mainloop()
