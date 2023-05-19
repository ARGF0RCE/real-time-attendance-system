import os
import re
import cv2
import requests
import datetime
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from model import *
from flask import current_app
import os
from werkzeug.utils import secure_filename
# from sshtunnel import SSHTunnelForwarder
import base64
import numpy as np


# # SSH tunnel configuration
# SSH_USERNAME = 'postgres'
# SSH_PASSWORD = '1235Eight13Fib.!@#'
# SSH_HOST = '172-105-50-204.ip.linodeusercontent.com'
# SSH_PORT = 22

# PostgreSQL configuration
DB_USERNAME = 'postgres'
DB_PASSWORD = 'pwd'
DB_NAME = 'attendance_db'
DB_PORT = 5432

# # Set up an SSH tunnel
# server = SSHTunnelForwarder(
#     (SSH_HOST, SSH_PORT),
#     ssh_username=SSH_USERNAME,
#     ssh_password=SSH_PASSWORD,
#     remote_bind_address=('localhost', DB_PORT),
#     local_bind_address=('localhost', 0)  # Automatically assign a free local port
# )

# # Start the SSH tunnel
# server.start()

# # Update the database URI to use the SSH tunnel

# print(f"Binded Local port: {server.local_bind_port}")
# Define the allowed extensions for uploaded images
ALLOWED_EXTENSIONS = {'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@localhost:{DB_PORT}/{DB_NAME}'
db = SQLAlchemy(app)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    student_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(10), nullable=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_student', methods=['POST'])
def add_student():
    student_name = request.form['student_name']
    student_photo = request.files['student_photo']

    if student_photo and allowed_file(student_photo.filename):
        filename = secure_filename(student_photo.filename)
        image_path = os.path.join('./backend/images/', filename)
        student_photo.save(image_path)
    else:
        return jsonify({'result': 'error', 'message': 'Invalid file format'})

    # Add a new entry to the Attendance table for the new student
    attendance_entry = Attendance(date=datetime.date.today(), student_name=student_name, status='absent')
    db.session.add(attendance_entry)
    db.session.commit()

    return jsonify({'result': 'success'})

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    date_of_attendance = datetime.date.today().strftime('%Y-%m-%d')
    
    # Get the base64 image string from the request
    image_data = request.form['image_data']
    # Remove the header of the base64 string
    header, base64_image_data = image_data.split(',', 1)
    decoded_image = base64.b64decode(base64_image_data)

    # Convert the base64 string to an OpenCV image
    nparr = np.frombuffer(decoded_image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Recognize faces in the image
    recognized_students = recognize(img)

    # Loop through the list of students and update their attendance status
    for student_name in recognized_students:
        # Convert the date string to a datetime.date object
        date_of_attendance_obj = datetime.datetime.strptime(date_of_attendance, '%Y-%m-%d').date()

        # Create a new Attendance instance with the recognized student data
        attendance = Attendance(date=date_of_attendance_obj, student_name=student_name, status='Present')

        # Add the new Attendance instance to the database
        db.session.add(attendance)

    # Commit the changes to the database
    db.session.commit()

    return jsonify({'result': 'success'})

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    # Fetch attendance records from the database
    attendance_data = Attendance.query.all()
    attendance_list = [{'date': entry.date, 'student_name': entry.student_name, 'status': entry.status} for entry in attendance_data]

    return jsonify(attendance_list)
@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    try:
        # Delete all attendance records
        Attendance.query.delete()
        db.session.commit()
        return jsonify({'result': 'success'})
    except Exception as e:
        print(e)
        return jsonify({'result': 'error'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
