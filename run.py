import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from keras.models import  load_model
import cv2
import numpy as np
from gtts import gTTS
from pygame import mixer
import os
mixer.init()

class_name = [ 'A', 'B', 'C', 'D' , 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' , 'del' , 'space']

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load model
model = load_model("asl_model.hdf5")

offset = 20
imgSize = 300

folder = "test"
counter = 0
text =""
description = "Press (R) to restart. Press (S) to speak out the record."
start = time.time()
letter_old = ""
letter = ""
check_even= 0
interval = 1.2
while True:

    success, img = cap.read()
    img2 = img.copy()
    hands, img = detector.findHands(img)
    display_text = cv2.imread("blank.png")
    description_text = cv2.imread("blank2.png")
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        image = cv2.resize(imgWhite, dsize=(128, 128))
        image = image.astype('float')*1./255
        # Convert to tensor
        image = np.expand_dims(image, axis=0)

        # Predict
        predict = model.predict(image)
        
        # Add letters to text
        end = time.time()
        if((end - start >= interval)  and check_even == 0 ):
            start = time.time()
            letter  = class_name[np.argmax(predict[0])]
            check_even = 1 
        if ((end - start >= interval)  and check_even == 1):
            check_even =0 
            start = time.time()
            letter_old = class_name[np.argmax(predict[0])]
            if(letter_old == letter):
                if letter != "del" and letter != "space":
                    text += letter
                elif letter == "del" :
                    text = text[0:-1]
                else:
                    text += "_"
                                
        cv2.rectangle(img2, (x- 20, y - 20),
                                  (x + w + 20, y + h + 20),
                                  (255, 0, 255), 2)
        cv2.putText(img2, class_name[np.argmax(predict[0])], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 255, 0), 2)
        


    cv2.putText(display_text, text, (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(description_text, description, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)

    cv2.imshow("Image", img2)
    cv2.imshow("text", display_text)
    cv2.imshow("description",description_text)

    
    key = cv2.waitKey(1)
    text_ = text.replace("_",  " ")
    # Press "R" to restart
    if key == ord("r"):
        text = ""
        mixer.music.unload()
        os.remove("test.mp3")
    # Press "S" to speak out the text
    if key == ord("s"):   
        speak = gTTS(text_,lang='en',slow='false')   
        try:
            speak.save("test.mp3")
        except:
            pass
        mixer.music.load("test.mp3")
        mixer.music.play()
