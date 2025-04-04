from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk  # pip install pillow
from tkinter import messagebox
import mysql.connector  # pip install mysql-connector-python
import cv2  # pip install opencv-python
import os
import numpy as np
from time import strftime
from datetime import datetime
import face_recognition  # pip install face-recognition


from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os
import numpy as np
from time import strftime
from datetime import datetime
import face_recognition
import torch
import torchaudio
import sounddevice as sd
import wave
from scipy.spatial.distance import cosine
from speechbrain.inference import EncoderClassifier


class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")
        
        self.var_std_id = StringVar()
        self.var_std_name = StringVar()
        
    # Load the Speaker Recognition Model
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_model"
        )

        # Title
        title_lbl = Label(self.root, text="FACE RECOGNITION", font=("times new roman", 35, "bold"), bg="white", fg="green")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        # Image 1
        img_1 = Image.open(r"images/good.jpg")
        img_1 = img_1.resize((650, 830), Image.LANCZOS)
        self.photoimg_1 = ImageTk.PhotoImage(img_1)

        bg_img1 = Label(self.root, image=self.photoimg_1)
        bg_img1.place(x=0, y=55, width=650, height=700)

        # Image 2
        img_2 = Image.open(r"images/face.jpg")
        img_2 = img_2.resize((950, 700), Image.LANCZOS)
        self.photoimg_2 = ImageTk.PhotoImage(img_2)

        bg_img2 = Label(self.root, image=self.photoimg_2)
        bg_img2.place(x=650, y=55, width=950, height=700)

        # Button
        b1_1 = Button(bg_img2, text="Face Recognition", command=self.face_recog, cursor="hand2", font=("times new roman", 18, "bold"), bg="red", fg="white")
        b1_1.place(x=0, y=600, width=300, height=60)

    # ====================== Attendance =====================

    def mark_attendance(self, student_id, roll, name, department):
        try:
            with open("detail_saving.csv", "r+", newline="\n") as f:
                myDataList = f.readlines()
                name_list = [line.split(",")[0] for line in myDataList]

                if student_id not in name_list:
                    now = datetime.now()
                    date = now.strftime("%d/%m/%Y")
                    time = now.strftime("%H:%M:%S")
                    f.writelines(f"\n{student_id},{roll},{name},{department},{time},{date},Face Verified")

                    # Step 1: Show face recognition success message
                    messagebox.showinfo("Success", f"Face Recognized for {name}!\nNow take a voice sample to mark full attendance.")

                    # Step 2: Call voice recognition and ensure it belongs to the same person
                    voice_verified = self.recognize_speaker(self.var_std_id, name)


                    # Step 3: Only mark attendance if voice matches
                    if voice_verified:
                        f.writelines(f"\n{student_id},{roll},{name},{department},{time},{date},Fully Present")
                        messagebox.showinfo("Success", f"Final Attendance Marked for {name}!")
                    else:
                        messagebox.showwarning("Warning", "Voice not recognized! Attendance not fully marked.")

                    # Step 4: Exit system after voice verification (even if failed)
                    self.exit_system()

        except Exception as e:
            messagebox.showerror("Error", f"Error marking attendance: {e}")


    def exit_system(self):
        """Release webcam and close OpenCV window."""
        cv2.destroyAllWindows()
        exit()


#     # =================== Face Recognition ==========================
#     def face_recog(self):
#         """Perform face recognition using the webcam."""
#         # Load known face encodings and their corresponding IDs from the data folder
#         known_face_encodings = []
#         known_face_ids = []

#         try:
#             # Iterate through all .npy files in the data folder
#             for file in os.listdir("data"):
#                 if file.endswith(".npy"):
#                     # Extract student ID from the filename
#                     student_id = file.split(".")[1]

#                     # Load the face encoding from the .npy file
#                     face_encoding = np.load(os.path.join("data", file))
#                     known_face_encodings.append(face_encoding)
#                     known_face_ids.append(student_id)

#             if not known_face_encodings:
#                 messagebox.showerror("Error", "No face encodings found in the data folder. Please add face encodings first.")
#                 return

#         except Exception as e:
#             messagebox.showerror("Error", f"Error loading face encodings: {e}")
#             return

#         # Initialize webcam
#         video_cap = cv2.VideoCapture(1)  # Use index 0 for the default webcam

#         if not video_cap.isOpened():
#             messagebox.showerror("Error", "Unable to access the webcam. Please check your camera.")
#             return

#         # Confidence threshold (adjust as needed)
#         confidence_threshold = 0.35  # Faces with a distance greater than this are considered unknown

#         while True:
#             ret, frame = video_cap.read()
#             if not ret:
#                 messagebox.showerror("Error", "Unable to capture frame from webcam.")
#                 break

#             # Resize the frame for faster processing
#             small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#             # Convert the image from BGR to RGB
#             rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#             # Find all face locations and encodings in the current frame
#             face_locations = face_recognition.face_locations(rgb_small_frame)
#             face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#             for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#                 # Scale back the face locations to the original frame size
#                 top *= 4
#                 right *= 4
#                 bottom *= 4
#                 left *= 4

#                 # Compare the face with known faces
#                 face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#                 best_match_index = np.argmin(face_distances)
#                 best_match_distance = face_distances[best_match_index]

#                 if best_match_distance <= confidence_threshold:
#                     # Recognized face
#                     student_id = known_face_ids[best_match_index]

#                     try:
#                         # Fetch student details from the database
#                         conn = mysql.connector.connect(
#                             host="localhost",
#                             username="root",
#                            password="Maaz%2006",
#                             database="maazdb"
#                         )
#                         my_cursor = conn.cursor()
#                         my_cursor.execute("SELECT Name, Roll, Dep FROM students WHERE Student_id=%s", (student_id,))
#                         result = my_cursor.fetchone()

#                         if result:
#                             name, roll, dep = result

#                             # Mark attendance
#                             self.mark_attendance(student_id, roll, name, dep)

#                             # Draw a rectangle around the face
#                             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

#                             # Display the name and details
#                             cv2.putText(frame, f"ID: {student_id}", (left, top - 75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                             cv2.putText(frame, f"Roll: {roll}", (left, top - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                             cv2.putText(frame, f"Name: {name}", (left, top - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                             cv2.putText(frame, f"Department: {dep}", (left, top - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                         else:
#                             # Draw a rectangle around the face
#                             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#                             cv2.putText(frame, "Unknown Face", (left, top - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                     except Exception as e:
#                         messagebox.showerror("Database Error", f"Error fetching student details: {e}")
#                     finally:
#                         if conn.is_connected():
#                             my_cursor.close()
#                             conn.close()
#                 else:
#                     # Unknown face
#                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#                     cv2.putText(frame, "Unknown Face", (left, top - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

#             # Display the frame
#             cv2.imshow("Face Recognition", frame)

#             # Exit on pressing 'Enter'
#             if cv2.waitKey(1) == 13:
#                 break

#         # Release the webcam and close the window
#         video_cap.release()
#         cv2.destroyAllWindows()


    def face_recog(self):
        """Perform face recognition using the webcam and trigger voice recognition before closing."""
        known_face_encodings = []
        known_face_ids = []

        try:
            # Load face encodings
            for file in os.listdir("data"):
                if file.endswith(".npy"):
                    student_id = file.split(".")[1]
                    face_encoding = np.load(os.path.join("data", file))
                    known_face_encodings.append(face_encoding)
                    known_face_ids.append(student_id)

            if not known_face_encodings:
                messagebox.showerror("Error", "No face encodings found in the data folder. Please add face encodings first.")
                return

        except Exception as e:
            messagebox.showerror("Error", f"Error loading face encodings: {e}")
            return

        # Initialize webcam
        video_cap = cv2.VideoCapture(0)

        if not video_cap.isOpened():
            messagebox.showerror("Error", "Unable to access the webcam. Please check your camera.")
            return

        confidence_threshold = 0.35  # Adjust for better accuracy

        while True:
            ret, frame = video_cap.read()
            if not ret:
                messagebox.showerror("Error", "Unable to capture frame from webcam.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]

                if best_match_distance <= confidence_threshold:
                    student_id = known_face_ids[best_match_index]

                    try:
                        conn = mysql.connector.connect(
                            host="localhost",
                            username="root",
                           password="Maaz%2006",
                            database="maazdb"
                        )
                        my_cursor = conn.cursor()
                        my_cursor.execute("SELECT Name, Roll, Dep FROM students WHERE Student_id=%s", (student_id,))
                        result = my_cursor.fetchone()

                        if result:
                            name, roll, dep = result
                            self.mark_attendance(student_id, roll, name, dep)

                            # Step 1: **Face Recognized - Ask for Voice Sample**
                            messagebox.showinfo("Face Recognized", f"{name} (ID: {student_id}) recognized!\nNow take a voice sample.")

                            # Step 2: **Trigger Voice Recognition**
                            voice_verified = self.recognize_speaker(student_id, name)



                            # Step 3: **Final Attendance Confirmation**
                            if voice_verified:
                                messagebox.showinfo("Success", f"Final Attendance Marked for {name}!")
                                video_cap.release()
                                cv2.destroyAllWindows()
                                return  # **Exit after successful attendance**
                            else:
                                messagebox.showwarning("Warning", "Voice not recognized! Attendance not fully marked.")
                                video_cap.release()
                                cv2.destroyAllWindows()
                                return  # **Exit after unsuccessful voice verification**

                    except Exception as e:
                        messagebox.showerror("Database Error", f"Error fetching student details: {e}")
                    finally:
                        if conn.is_connected():
                            my_cursor.close()
                            conn.close()

            # Display the frame until recognition is done
            cv2.imshow("Face Recognition", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        video_cap.release()
        cv2.destroyAllWindows()



    def record_audio(self, filename, duration=3, fs=16000):
        print("Recording... Please speak now!")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        
        wavefile = wave.open(filename, 'wb')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(fs)
        wavefile.writeframes(audio.tobytes())
        wavefile.close()
        print(f"Recording saved as {filename}")

    def extract_voiceprint(self, audio_path):
        signal, fs = torchaudio.load(audio_path)
        embeddings = self.classifier.encode_batch(signal)
        return embeddings.squeeze().detach().numpy()

    
    def recognize_speaker(self, student_id, student_name, threshold=0.35):
        """Recognize the speaker and ensure voice matches the recognized face."""
        test_wav = "test_audio.wav"
        self.record_audio(test_wav)

        test_voiceprint = self.extract_voiceprint(test_wav)

        # Ensure only the recognized student's voiceprint is checked
        expected_voiceprint_path = os.path.join("voiceprints", f"{student_id}_{student_name}.npy")
        
        if not os.path.exists(expected_voiceprint_path):
            messagebox.showwarning("Warning", f"No voice sample found for {student_name}. Please enroll first.")
            return False

        stored_voiceprint = np.load(expected_voiceprint_path)
        similarity = 1 - cosine(test_voiceprint, stored_voiceprint)

        print(f"Comparing with {student_name}: Similarity = {similarity:.4f}")

        if similarity >= threshold:
            messagebox.showinfo("Success", f"Voice Verified for {student_name} (Similarity: {similarity:.4f})")
            return True
        else:
            messagebox.showwarning("Failure", "Voice not recognized! Attendance not marked.")
            return False
 
    




if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()