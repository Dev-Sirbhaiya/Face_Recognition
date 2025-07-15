import face_recognition as fc
import cv2
from fer import FER
from pathlib import Path

detector = FER(mtcnn=True)
KNOWN_DIR = Path(__file__).parent / "Face Detection and Recognition Project"#Make sure to set the correct path to your known faces directory

known_enc = []
known_names = []

for img in KNOWN_DIR.glob("*.*"):#This section loads known faces from the specified directory and makes encoding of them to compare with unknown faces
    image_bgr = cv2.imread(str(img))
    if image_bgr is None:
        continue
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_location = fc.face_locations(image)
    image_encoding = fc.face_encodings(image, image_location)
    if len(image_encoding) == 0:
        continue
    known_enc.append(image_encoding[0])#encodings are appended to the list
    known_names.append(img.stem)

cap = cv2.VideoCapture(0)#Takes an image from webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    unknown_bgr = frame.copy()
    unknown = cv2.cvtColor(unknown_bgr, cv2.COLOR_BGR2RGB)
    unknown_locations = fc.face_locations(unknown)
    unknown_encodings = fc.face_encodings(unknown, unknown_locations)# This section detects faces in the frame and encodes them

    if len(unknown_encodings) == 0:
        continue

    for enc in unknown_encodings:
        distances = fc.face_distance(known_enc, enc)
        value = distances.argmin()#Compares to find the closest match
        result_name = known_names[value]

    results = fc.compare_faces(known_enc, enc, 0.5)

    if results[value]:
        feeling = detector.top_emotion(unknown)
        print(result_name, "is", feeling[0])
    else:
        print("No Match")

    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
