from cvzone.FaceDetectionModule import FaceDetector
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize face detector
detector = FaceDetector()

while True:
    success, img = cap.read()

    # Detect faces
    img, bboxs = detector.findFaces(img)

    # If faces are detected
    if bboxs:
        # Each bbox contains: "id", "bbox", "score", "center"
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    # Show the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
