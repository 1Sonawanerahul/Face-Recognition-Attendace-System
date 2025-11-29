# Face Recognition Attendance System 🎭📊

A comprehensive and advanced face recognition-based attendance system built with Python and OpenCV. This system automatically marks attendance by recognizing registered students' faces in real-time using webcam feed.

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **👤 Student Registration** - Register new students with unique IDs and face data
- **🎯 Real-time Face Recognition** - Multiple face detection and recognition
- **📊 Automated Attendance** - Automatic attendance marking with timestamps
- **🆔 Unique ID System** - Auto-generated student IDs (STU001, STU002, etc.)
- **💾 Database Management** - JSON-based student database and CSV attendance records
- **📷 Multiple Photo Capture** - Capture 10 photos per student for better accuracy
- **🔍 Multiple Face Support** - Detect and recognize multiple faces simultaneously
- **📈 Attendance Reports** - View and export attendance records
- **🎮 User-friendly Menu** - Easy-to-use console interface

## 🚀 Quick Start

### Prerequisites

- Python 3.6 or higher
- Webcam
- Required Python packages

## Installation

1. **Clone the repository**
   ```bash
      git clone https://github.com/yourusername/face-recognition-attendance-system.git
   
      cd face-recognition-attendance-system

2. **install required packages**
   ```bash
      pip install opencv-contrib-python numpy pandas
   
3. **Run the system**
   ```bash
      python advanced_attendance.py


## 🎮 How to Use

### 1. Student Registration
   Select option  - 1 from main menu

   Enter student name

   System generates unique ID (e.g., STU001)

   Capture 10 face photos from different angles

   Student data saved to database


### 2. Take Attendance
   Select option 2 from main menu

   System trains with registered students

   Webcam starts for real-time face recognition

   Recognized students are automatically marked present

   Multiple students can be detected simultaneously


### 3. View Records
   Option 3: View all registered students

   Option 4: View attendance records

   Option 5: Exit system
   


## 🛠️ Technical Details

### Technologies Used

   OpenCV - Face detection and recognition

   LBPH Algorithm - Local Binary Patterns Histograms for face recognition

   Haar Cascades - Face detection
   
   JSON - Student database storage

   Pandas - Attendance records management

   NumPy - Numerical computations

### Key Components

   face_cascade - Haar cascade for face detection

   recognizer - LBPH face recognizer

   students_database.json - Stores student information

   attendance_records.csv - Stores attendance data

   face_data/ - Directory for student face images



 ## 📊 Sample Data
 
#### Student Database (students_database.json)
      ```json

      {
       "STU001": {
           "name": "Rahul",
           "registration_date": "25/11/2025 22:07:43",
           "photos_count": 10
                }
      }

#### Attendance Records (attendance_records.csv)

   Student_ID,Name,Time,Date
   
STU001,Rahul,22:08:00,25/11/2025

STU002,Vedansh,22:10:33,25/11/2025



## 🎯 Accuracy & Performance

### Face Detection: Haar cascades with 1.3 scale factor and 5 min neighbors

   Recognition Confidence: Threshold set at 70 (lower is better)

   Multiple Faces: Can detect and recognize multiple faces simultaneously

   Training: Uses 10 photos per student for robust recognition

## 🔧 Customization

   #### Adjust Recognition Sensitivity
   
   Modify the confidence threshold in start_attendance_system() function:

   ```python
      if confidence < 70:  # Lower value = more strict recognition

