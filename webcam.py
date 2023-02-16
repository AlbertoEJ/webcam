import streamlit as st
import cv2
import numpy as np





import face_recognition
from sklearn import svm, neighbors
import os
import cv2
import pickle
# Training the SVC classifier

def train( X, y, model_save_path, n_neighbors=2, knn_algo='ball_tree',):
     # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('train_dir/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("train_dir/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("train_dir/" + person + "/" + person_img)
        face = cv2.resize(face, (0, 0), fx=0.25, fy=0.25)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face, known_face_locations=face_bounding_boxes)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")



# Create and train the SVC classifier
#clf = svm.SVC(gamma='scale')
#clf.fit(encodings,names)
model_save_path = 'knn.clf'

clf = train(encodings, names, model_save_path)
distance_threshold = 0.6


frameWidth = 640
frameHeight = 480
video_capture = cv2.VideoCapture(0)
video_capture.set(3, frameWidth)
video_capture.set(4, frameHeight)
video_capture.set(10,150)

# # Load a sample picture and learn how to recognize it.
# eduardo_image = face_recognition.load_image_file(f'faces/eduardo.jpg')
# eduardo_face_encoding = face_recognition.face_encodings(eduardo_image)[0]

# # Load a second sample picture and learn how to recognize it.
# sansa_image = face_recognition.load_image_file(f'faces/sansa.jpg')
# sansa_face_encoding = face_recognition.face_encodings(sansa_image)[0]

# # Create arrays of known face encodings and their names
# known_face_encodings = [
#     eduardo_face_encoding,
#     sansa_face_encoding
# ]

# #print(known_face_encodings)
# known_face_names = [
#     "Eduardo",
#     "Sansa"
# ]
#print(known_face_names)
# Initialize some variables


face_locations = []
face_encodings = []
#face_names = []
process_this_frame = True

while video_capture.isOpened():
    success, frame = video_capture.read()
    if success:

        # Only process every other frame of video to save time
        if process_this_frame:

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)

            if len(face_locations) != 0:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=face_locations)
                
                #face_names = []

                # Use the KNN model to find the best matches for the test face
                closest_distances = clf.kneighbors(face_encodings, n_neighbors=1)
                are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

                # Predict classes and remove classifications that aren't within the threshold
                res = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(clf.predict(face_encodings), face_locations, are_matches)]
                #for pred, loc, rec in zip(clf.predict(face_encodings), face_locations, are_matches):
                #    if rec:
                #        face_names.append(pred)
                    #(pred, loc)
                #    else:
                #        face_names.append('Desconocido')
            else:
                res = []  

        process_this_frame = not process_this_frame

            # Display the results
        for name, (top, right, bottom, left)  in res:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()







# ////////////////////////////////////////////////////////////////////////////////////////////////



