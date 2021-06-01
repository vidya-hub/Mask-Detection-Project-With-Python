from numpy.linalg import norm
import cv2
import numpy as np
import mediapipe as mp
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

model = tensorflow.keras.models.load_model('keras_model.h5')

np.set_printoptions(suppress=True)
size = (224, 224)


def predict(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5,)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(imgRgb)
    if (results.detections) != None:
        for face in (results.detections):
            # mp_drawing.draw_detection(img, face)
            face_data = face.location_data.relative_bounding_box
            h, w, c = img.shape
            x, y, width, height = int(face_data.xmin*w), int(face_data.ymin*h), \
                int(face_data.width*w), int(face_data.height*h)
            # print(x, y)

            if (x > 0 and y > 0):
                face_rect = img[y:y+height, x:x+width]
                faceGray = cv2.cvtColor(face_rect, cv2.COLOR_BGR2GRAY)
                pred = predict(faceGray)
                if (np.argmax(pred) == 0):
                    cv2.putText(img, f"Mask On", (50, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                else:
                    cv2.putText(img, f"Mask Off", (50, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                cv2.imshow("Face", face_rect)
            cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 1,)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
