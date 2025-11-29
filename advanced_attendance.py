import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import json

print("🚀 Advanced Face Recognition Attendance System Starting...")

# Face detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Database files
STUDENTS_FILE = "students_database.json"
ATTENDANCE_FILE = "attendance_records.csv"

def load_students_database():
    """Load students database from JSON file"""
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_students_database(database):
    """Save students database to JSON file"""
    with open(STUDENTS_FILE, 'w') as f:
        json.dump(database, f, indent=4)

def generate_student_id(database):
    """Generate unique student ID"""
    if not database:
        return "STU001"
    
    existing_ids = [int(id_[3:]) for id_ in database.keys() if id_.startswith('STU')]
    if not existing_ids:
        return "STU001"
    
    new_id = max(existing_ids) + 1
    return f"STU{new_id:03d}"

def register_new_student():
    """Register a new student with face data"""
    database = load_students_database()
    
    print("\n" + "="*50)
    print("         NEW STUDENT REGISTRATION")
    print("="*50)
    
    # Get student details
    name = input("Enter student name: ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    # Check if name already exists
    for student_id, details in database.items():
        if details['name'].lower() == name.lower():
            print(f"❌ Student '{name}' already registered with ID: {student_id}")
            return
    
    # Generate student ID
    student_id = generate_student_id(database)
    
    print(f"\n📝 Registering: {name}")
    print(f"🎫 Assigned ID: {student_id}")
    
    # Take photos for training
    print("\n📸 Now taking face photos for registration...")
    photos_taken = take_student_photos(student_id, name)
    
    if photos_taken > 0:
        # Add to database
        database[student_id] = {
            'name': name,
            'registration_date': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'photos_count': photos_taken
        }
        save_students_database(database)
        print(f"\n✅ Student registered successfully!")
        print(f"   Name: {name}")
        print(f"   ID: {student_id}")
        print(f"   Photos taken: {photos_taken}")
    else:
        print("❌ Registration failed! No photos taken.")

def take_student_photos(student_id, name):
    """Take multiple photos for a student"""
    student_folder = f"face_data/{student_id}_{name}"
    if not os.path.exists(student_folder):
        os.makedirs(student_folder)
    
    cap = cv2.VideoCapture(0)
    count = 0
    max_photos = 10
    
    print(f"\n📸 Taking photos for {name} ({student_id})")
    print("Instructions:")
    print("1. Face the camera directly")
    print("2. 's' - Take photo")
    print("3. 'q' - Finish registration")
    print(f"4. Take {max_photos} photos from different angles")
    
    while count < max_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Photos: {count}/{max_photos}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Student: {name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"ID: {student_id}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(f'Registration - {name} - Press S to capture, Q to finish', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(faces) > 0:
                count += 1
                filename = f"{student_folder}/photo_{count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ Photo {count} saved")
            else:
                print("❌ No face detected!")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return count

def train_face_recognizer():
    """Train the face recognizer with all registered students"""
    print("\n📚 Training face recognizer with all students...")
    
    faces = []
    ids = []
    id_to_info = {}
    
    database = load_students_database()
    if not database:
        print("❌ No students registered! Please register students first.")
        return False, {}
    
    # Create a mapping for ID to numeric ID
    id_mapping = {}
    current_numeric_id = 0
    
    for student_id, details in database.items():
        student_folder = f"face_data/{student_id}_{details['name']}"
        
        if os.path.exists(student_folder):
            for filename in os.listdir(student_folder):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(student_folder, filename)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        for (x, y, w, h) in detected_faces:
                            face_region = gray[y:y+h, x:x+w]
                            faces.append(face_region)
                            
                            if student_id not in id_mapping:
                                id_mapping[student_id] = current_numeric_id
                                id_to_info[current_numeric_id] = {
                                    'student_id': student_id,
                                    'name': details['name']
                                }
                                current_numeric_id += 1
                            
                            ids.append(id_mapping[student_id])
    
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        print(f"✅ Training complete! {len(id_mapping)} students trained")
        return True, id_to_info
    else:
        print("❌ No face data found for training!")
        return False, {}

def mark_attendance(student_id, name):
    """Mark attendance for a student"""
    today = datetime.now().strftime('%d/%m/%Y')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Create or load attendance file
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=['Student_ID', 'Name', 'Time', 'Date'])
    
    # Check if already marked today
    today_attendance = df[(df['Student_ID'] == student_id) & (df['Date'] == today)]
    
    if today_attendance.empty:
        new_entry = {
            'Student_ID': student_id,
            'Name': name,
            'Time': current_time,
            'Date': today
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"✅ Attendance marked: {name} ({student_id}) at {current_time}")
        return True
    else:
        print(f"⚠️ Already marked: {name} ({student_id})")
        return False

def view_registered_students():
    """View all registered students"""
    database = load_students_database()
    
    if not database:
        print("\n❌ No students registered yet!")
        return
    
    print("\n" + "="*50)
    print("         REGISTERED STUDENTS")
    print("="*50)
    
    for i, (student_id, details) in enumerate(database.items(), 1):
        print(f"{i}. ID: {student_id}")
        print(f"   Name: {details['name']}")
        print(f"   Registered: {details['registration_date']}")
        print(f"   Photos: {details.get('photos_count', 0)}")
        print()

def start_attendance_system():
    """Start the main attendance system"""
    print("\n🎥 Starting Attendance System...")
    
    training_success, id_to_info = train_face_recognizer()
    if not training_success:
        return
    
    cap = cv2.VideoCapture(0)
    marked_today = set()
    
    print("\n📋 Attendance System Active!")
    print("Instructions:")
    print("- Multiple faces can be detected at once")
    print("- Recognized students will be marked automatically")
    print("- Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error!")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            numeric_id, confidence = recognizer.predict(face_roi)
            
            if confidence < 70 and numeric_id in id_to_info:
                # Recognized student
                student_info = id_to_info[numeric_id]
                student_id = student_info['student_id']
                name = student_info['name']
                
                color = (0, 255, 0)  # Green
                status = f"{name} ({student_id})"
                
                # Mark attendance
                today = datetime.now().strftime('%d/%m/%Y')
                unique_key = f"{student_id}_{today}"
                
                if unique_key not in marked_today:
                    mark_attendance(student_id, name)
                    marked_today.add(unique_key)
            else:
                # Unknown person
                color = (0, 0, 255)  # Red
                status = "Unknown"
            
            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Display frame count
        cv2.putText(frame, "Multiple Face Detection Active", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Advanced Attendance System - Multiple Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Attendance system closed!")

def main_menu():
    """Main menu system"""
    while True:
        print("\n" + "="*50)
        print("    ADVANCED FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*50)
        print("1. Register New Student")
        print("2. Start Attendance System")
        print("3. View Registered Students")
        print("4. View Attendance Records")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            register_new_student()
        elif choice == '2':
            start_attendance_system()
        elif choice == '3':
            view_registered_students()
        elif choice == '4':
            if os.path.exists(ATTENDANCE_FILE):
                df = pd.read_csv(ATTENDANCE_FILE)
                if not df.empty:
                    print("\n📊 Attendance Records:")
                    print(df)
                else:
                    print("❌ No attendance records found!")
            else:
                print("❌ No attendance file found!")
        elif choice == '5':
            print("👋 Thank you for using the system!")
            break
        else:
            print("❌ Invalid choice! Please try again.")

if __name__ == "__main__":
    # Create necessary folders
    if not os.path.exists("face_data"):
        os.makedirs("face_data")
    
    main_menu()