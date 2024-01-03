import cv2
import os
import numpy as np

def center_crop(img, x, y, w, h):
    center_x, center_y = x + w//2, y + h//2
    size = min(max(w, h), min(img.shape[0], img.shape[1]))
    top_y = max(center_y - size//2, 0)
    bottom_y = top_y + size
    left_x = max(center_x - size//2, 0)
    right_x = left_x + size

    return img[top_y:bottom_y, left_x:right_x]

def adjust_brightness_contrast(img):
    # Convert image to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Compute the mean of the Y channel
    y_channel = img_yuv[:,:,0]
    mean_y = np.mean(y_channel)

    # Adjust the Y channel based on the mean value to normalize brightness
    factor = 127 / mean_y
    adjusted_y = np.clip(y_channel * factor, 0, 255).astype(np.uint8)

    img_yuv[:,:,0] = adjusted_y

    # Convert back to BGR color space
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def process_image(img_path, face_cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    centered_face = center_crop(img, x, y, w, h)
    face_resized = cv2.resize(centered_face, (48, 48))
    adjusted_img = adjust_brightness_contrast(face_resized)

    return adjusted_img

def process_folder(folder_path, face_cascade):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                processed_img = process_image(img_path, face_cascade)

                if processed_img is not None:
                    new_path = img_path.replace(folder_path, folder_path + '_processed')
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    cv2.imwrite(new_path, processed_img)

face_cascade = cv2.CascadeClassifier('/Users/markusbaumann/emotionrecognition/haarcascade_frontalface_default.xml')
main_folder = '/Users/markusbaumann/Desktop/test/test'
emotions = ['angry', 'sad', 'happy', 'fear', 'disgust', 'surprise']

for emotion in emotions:
    process_folder(os.path.join(main_folder, emotion), face_cascade)
