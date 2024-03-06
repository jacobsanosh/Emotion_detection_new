import os
import cv2

input_directory = "./new_images"
output_folder = "./"

def process_and_save_faces(input_image_path, output_folder, j):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    expression = input_image_path.split("_")[-1].split(".")[0].upper()
    expression_folder = os.path.join(output_folder, expression)
    os.makedirs(expression_folder, exist_ok=True)
    print(expression)
    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (48, 48))
        output_path = os.path.join(expression_folder, f"{j+1}.jpg")
        # Save the image with higher quality (95 out of 100)
        cv2.imwrite(output_path, resized_face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

j = 40000
# for filename in os.listdir(input_directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         input_image_path = os.path.join(input_directory, filename)
#         process_and_save_faces(input_image_path, output_folder, j)
#         j += 1

print(len(os.listdir('./ANG')))
print(len(os.listdir('./DIS')))
print(len(os.listdir('./FER')))
print(len(os.listdir('./HPY')))
print(len(os.listdir('./NUT')))
print(len(os.listdir('./SAD')))
print(len(os.listdir('./SUR')))