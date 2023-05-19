import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from architecture import InceptionResNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Paths
TEST_DATA_DIR = './backend/images'
FACENET_MODEL_PATH = './backend/facenet_keras_weights_1.h5'

# Load MTCNN for face detection
detector = MTCNN()

# Load FaceNet model for generating embeddings
facenet_model = InceptionResNetV2()
facenet_model.load_weights(FACENET_MODEL_PATH)

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255
    return img

def get_embedding(img_path):
    img = preprocess_image(img_path)
    return facenet_model.predict(img)

# Load test data and generate embeddings
X_test, y_test = [], []
for img_name in os.listdir(TEST_DATA_DIR):
    img_path = os.path.join(TEST_DATA_DIR, img_name)
    label = os.path.splitext(img_name)[0]
    embedding = get_embedding(img_path)
    X_test.append(embedding)
    y_test.append(label)

X_test = np.asarray(X_test).squeeze()
y_test = np.asarray(y_test)

# Normalize embeddings
normalizer = Normalizer(norm='l2')
X_test_norm = normalizer.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Train a KNN classifier on the embeddings
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_test_norm, y_test_encoded)

# Evaluate the classifier's accuracy
y_pred = classifier.predict(X_test_norm)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
