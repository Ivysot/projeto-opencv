import cv2

arquivoImportar = cv2.CascadeClassifier("opencv\data\haarcascades\haarcascade_frontalface_default.xml")

imgLeitura = cv2.imread("imagens-teste\img2.jpg")

'imgCinza = cv2.cvtColor(imgLeitura, cv2.COLOR_BGR2GRAY)'

faces = arquivoImportar.detectMultiScale(imgLeitura)

print(faces)

for(x,y,l,a) in faces:
    cv2.rectangle(imgLeitura, (x,y), (x+l,y+a), (0,255,0), 2)

cv2.imshow("Faces", imgLeitura)

cv2.waitKey()