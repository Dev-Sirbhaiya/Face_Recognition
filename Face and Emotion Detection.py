import face_recognition as fc
import cv2
from fer import FER
from pathlib import Path

def load_known_faces(known_dir_path):
    known_enc = []
    known_names = []
    for img in known_dir_path.glob("*.*"):
        image_bgr = cv2.imread(str(img))
        if image_bgr is None:
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_location = fc.face_locations(image)
        image_encoding = fc.face_encodings(image, image_location)
        if len(image_encoding) == 0:
            continue
        known_enc.append(image_encoding[0])
        known_names.append(img.stem)
    return known_enc, known_names

def recognize_and_emote(frame, known_enc, known_names, detector):
    unknown_bgr = frame.copy()
    unknown = cv2.cvtColor(unknown_bgr, cv2.COLOR_BGR2RGB)
    unknown_locations = fc.face_locations(unknown)
    unknown_encodings = fc.face_encodings(unknown, unknown_locations)

    if len(unknown_encodings) == 0:
        return frame

    for enc in unknown_encodings:
        distances = fc.face_distance(known_enc, enc)
        value = distances.argmin()
        result_name = known_names[value]
        results = fc.compare_faces(known_enc, enc, 0.5)

        if results[value]:
            feeling = detector.top_emotion(unknown)
            print(result_name, "is", feeling[0])
        else:
            print("No Match")

    return frame

def main():
    KNOWN_DIR = Path(__file__).parent / "Face Detection and Recognition Project"
    detector = FER(mtcnn=True)
    known_enc, known_names = load_known_faces(KNOWN_DIR)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognize_and_emote(frame, known_enc, known_names, detector)

        cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
