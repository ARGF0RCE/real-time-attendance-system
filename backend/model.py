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

    def recognize(img):
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

        # Train a k-NN classifier on the embeddings
        X_train = np.array(embeddings)
        y_train = np.array(students_list)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

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
