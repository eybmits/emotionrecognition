import cv2
import os

def process_image(img_path, face_cascade):
    # Read the image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Focus on the first face detected
    x, y, w, h = faces[0]

    # Crop and resize image
    face_centered = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_centered, (48, 48))

    return face_resized

def process_folder(folder_path, face_cascade):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                processed_img = process_image(img_path, face_cascade)

                if processed_img is not None:
                    # Create new path in the 'processed' directory
                    new_path = img_path.replace(folder_path, folder_path + '_processed')
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    cv2.imwrite(new_path, processed_img)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Main folder
main_folder = 'test'

# Process each emotion subfolder
emotions = ['angry', 'sad', 'happy', 'fear', 'disgust', 'surprise']
for emotion in emotions:
    process_folder(os.path.join(main_folder, emotion), face_cascade)
