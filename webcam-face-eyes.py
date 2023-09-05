import cv2

webcam = cv2.VideoCapture(0)
classifierFace = cv2.CascadeClassifier('opencv\data\haarcascades\haarcascade_frontalface_default.xml')
classifierEyes = cv2.CascadeClassifier('opencv\data\haarcascades\haarcascade_eye.xml')

while True:
    camera, frame = webcam.read()
    escalaCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection = classifierFace.detectMultiScale(escalaCinza)
    for(x,y,l,a) in detection:
        cv2.rectangle(frame, (x,y), (x+l, y+a), (225, 0, 0), 2)
        eyes = frame[y:y+a,x:x + l]
        grayEyes = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        eyesLocation = classifierEyes.detectMultiScale(grayEyes)
        for(ex,ey,el,ea) in eyesLocation:
            cv2.rectangle(eyes, (ex,ey),(ex+el, ey+ea), (0,225,0), 2)
    cv2.imshow("Minha detecção", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
