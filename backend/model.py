# To train a face recognition model on the images of the three students, 
# you can use the FaceNet model as a feature extractor and use a classifier like k-Nearest Neighbors (k-NN) for recognition.
# training and testing the model on a live video stream
# The model is trained on the images of three students and then tested on a live video stream to recognize the faces of the students.
import tensorflow as tf
with tf.device('/device:CPU:0'):
    import cv2
    import numpy as np
    import os
    from mtcnn import MTCNN
    from architecture import InceptionResNetV2
    from sklearn.neighbors import KNeighborsClassifier

    knn = None
    
    # Load the model
    model = InceptionResNetV2()
    model.load_weights('./backend/facenet_keras_weights_1.h5')

    # Load the pre-trained MTCNN model

    detector = MTCNN()

    # Function to preprocess the input images
    def preprocess(img):
        img = cv2.resize(img, (160, 160))
        img = img.astype('float32')
        img = (img - 127.5) / 128
        return np.expand_dims(img, axis=0)

    # Function to extract the face embeddings
    def get_embedding(model, face):
        face = preprocess(face)
        embedding = model.predict(face)
        return embedding[0]

    # Function to train on the images of the three students
    def train():
        # Load and preprocess the image of the three students
        students = os.listdir('./backend/images/')
        students_list = []
        # Iterate through the files and extract the filename without the extension
        for file in students:
            filename_without_ext, _ = os.path.splitext(file)
            students_list.append(filename_without_ext)
        # print(students_list)
        embeddings = []

        for student in students_list:
            img = cv2.imread(f'./backend/images/{student}.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = detector.detect_faces(img)[0]['box']
            face = img[max(0, face[1]):min(face[1]+face[3], img.shape[0]), max(0, face[0]):min(face[0]+face[2], img.shape[1])]
            print(face.shape)
            embedding = get_embedding(model, face)
            embeddings.append(embedding)
        return embeddings, students_list
    
    # Initialize the k-NN classifier
    def init_knn_classifier():
        global knn
        embeddings, students_list = train()
        X_train = np.array(embeddings)
        y_train = np.array(students_list)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

    # Call this function at the beginning to initialize the k-NN classifier
    init_knn_classifier()
    
    def recognize(img):
        # Detect faces in the input image
        global knn
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        predictions = []
        for face in faces:
            face_coordinates = face['box']
            cropped_face = img[max(0, face_coordinates[1]):min(face_coordinates[1]+face_coordinates[3], img.shape[0]), max(0, face_coordinates[0]):min(face_coordinates[0]+face_coordinates[2], img.shape[1])]
            embedding = get_embedding(model, cropped_face)
            embedding = np.array([embedding])
            pred = knn.predict(embedding)[0]
            predictions.append(pred)
        return predictions
    

    def recognize_test(img):
        # Detect faces in the input image
        faces = detector.detect_faces(img)

        # Train the model with the images of the three students
        embeddings, students_list = train()

        # Train a k-NN classifier on the embeddings
        X_train = np.array(embeddings)
        y_train = np.array(students_list)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        # Iterate over the detected faces
        for face in faces:
            x, y, w, h = face['box']
            cropped_face = img[y:y+h, x:x+w]

            # Compute the embedding for the face
            face_embedding = get_embedding(model, cropped_face)

            # Predict the label of the face using the k-NN classifier
            student_label = knn.predict(face_embedding.reshape(1, -1))[0]

            # Draw a bounding box around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Write the student's name below the bounding box
            cv2.putText(img, student_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return img