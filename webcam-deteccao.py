import cv2

webcam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('opencv\data\haarcascades\haarcascade_frontalface_default.xml')

while True:
    camera, frame = webcam.read()
    escalaCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection = classifier.detectMultixScale(escalaCinza)
    for(x,y,l,a) in detection:
        cv2.rectangle(frame, (x,y), (x+l, y+a), (225, 0, 0), 2)
    cv2.imshow("Minha detecção", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
