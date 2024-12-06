import cv2

# Load the pre-trained LBPH face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/trained_model2.yml')

# Dictionary containing paths to Haar cascade classifiers
cascadePaths = {
    "frontalface": "haarcascade_frontalface_alt.xml",
    "lefteye_2splits": "haarcascade_lefteye_2splits.xml",
    "mcs_eyepair_big": "haarcascade_mcs_eyepair_big.xml",
    "mcs_eyepair_small": "haarcascade_mcs_eyepair_small.xml",
    "mcs_leftear": "haarcascade_mcs_leftear.xml",
    "mcs_lefteye": "haarcascade_mcs_lefteye.xml",
    "mcs_mouth": "haarcascade_mcs_mouth.xml",
    "mcs_nose": "haarcascade_mcs_nose.xml",
    "mcs_rightear": "haarcascade_mcs_rightear.xml",
    "mcs_righteye": "haarcascade_mcs_righteye.xml",
    "righteye_2splits": "haarcascade_righteye_2splits.xml"
}

# Load Haar cascade classifiers
faceCascade = cv2.CascadeClassifier(cascadePaths["frontalface"])
lefteyeCascade = cv2.CascadeClassifier(cascadePaths["lefteye_2splits"])
eyepairBigCascade = cv2.CascadeClassifier(cascadePaths["mcs_eyepair_big"])
eyepairSmallCascade = cv2.CascadeClassifier(cascadePaths["mcs_eyepair_small"])
leftearCascade = cv2.CascadeClassifier(cascadePaths["mcs_leftear"])
lefteyeCascade = cv2.CascadeClassifier(cascadePaths["mcs_lefteye"])
mouthCascade = cv2.CascadeClassifier(cascadePaths["mcs_mouth"])
noseCascade = cv2.CascadeClassifier(cascadePaths["mcs_nose"])
rightearCascade = cv2.CascadeClassifier(cascadePaths["mcs_rightear"])
righteyeCascade = cv2.CascadeClassifier(cascadePaths["mcs_righteye"])
righteyeCascade = cv2.CascadeClassifier(cascadePaths["righteye_2splits"])

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['None', 'Vamsi']  # Add your dataset names here

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Perform face recognition
        id, confidence = recognizer.predict(roi_gray)

        if confidence < 100:
            if id < len(names):
                id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        # Detect facial features
        lefteyes = lefteyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in lefteyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        eyepairs_big = eyepairBigCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyepairs_big:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        eyepairs_small = eyepairSmallCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyepairs_small:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

        leftears = leftearCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in leftears:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

        mouth = mouthCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in mouth:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)

        noses = noseCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in noses:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        rightears = rightearCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in rightears:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (128, 128, 128), 2)

        righteyes = righteyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in righteyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 128, 128), 2)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
