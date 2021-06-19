#python drowniness_yawn.py --webcam webcam_index
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from keras.models import load_model
from pygame import mixer
from skimage.transform import resize 
from tkinter import *
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


root = Tk()
# ---------------- Main Window---------------------------#
root.title("Python Application")

root.resizable(0, 0)

def startApp():
    print("-> Loading the predictor and detector...")

    mixer.init()
    sound = mixer.Sound('alarm.wav')

    #detector = dlib.get_frontal_face_detector()
    detector = cv2.CascadeClassifier("haar cascade files\haarcascade_frontalface_default.xml")    #Faster but less accurate
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

    eyeModel = load_model('models/cnncat2.h5')
    mouthModel = load_model('models/alexnet.h5')

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count = 0
    score = 0
    yscore = 0
    thicc = 2
    rpred = [99]
    lpred = [99]

    lbl1 = ""
    lbl2 = ""
    lbl3 = ""
    print("-> Starting Video Stream")
    vs = VideoStream(src=0).start()
    #vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
    time.sleep(1.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        #rects = detector(gray, 0)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        # yawn detection
        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
            
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            distance = lip_distance(shape)

            lip = shape[48:60]
            
            mask = np.zeros(gray.shape,dtype=np.uint8)
            input_img = np.empty((1, 224, 224, 1), dtype=np.float32)

            mask = cv2.drawContours(mask, [lip], -1, (255 , 255 , 255),thickness=cv2.FILLED)
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            mask = mask/1.0
            img = resize(mask, output_shape=(224, 224, 1), preserve_range=True)
            #cv2.imshow('masked image',img)
            input_img[0] = img
            res = mouthModel.predict(input_img)
            if res[0][0] > res[0][1]:
                lbl1 = 'Yawn'
                yscore = yscore + 1
            else:
                lbl1 = 'Normal'
                yscore = yscore - 1
            cv2.putText(frame, lbl1, (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            break

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y+h, x:x+w]
            count = count+1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye/255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = eyeModel.predict(r_eye)
            if(rpred[0][0] < rpred[0][1]):
                lbl2 = 'Open'
            else:
                lbl2 = 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y+h, x:x+w]
            count = count+1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye/255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = eyeModel.predict(l_eye)
            if(lpred[0][0] < lpred[0][1]):
                lbl3 = 'Open'
            else:
                lbl3 = 'Closed'
            break

        if(lbl2 == "Closed" and lbl3 == "Closed"):
            score = score+1
            cv2.putText(frame, "Closed", (10, height-20), font,
                        1, (255, 255, 255), 1, cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score = score-1
            cv2.putText(frame, "Open", (10, height-20), font,
                        1, (255, 255, 255), 1, cv2.LINE_AA)

        if(yscore < 0):
            yscore = 0

        if(score < 0):
            score = 0
        cv2.putText(frame, 'Score:'+str(score)+' '+str(yscore), (100, height-20),
                    font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if(score > 15 or yscore > 15):
            # person is feeling sleepy so we beep the alarm
            #cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()

            except:  # isplaying = False
                pass
            if(thicc < 16):
                thicc = thicc+2
            else:
                thicc = thicc-2
                if(thicc < 2):
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)


        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

# Main window
# Frames
Top = Frame(root, bd=2,  relief=RIDGE)
Top.pack(side=TOP, fill=X)
Form = Frame(root, height=200)
Form.pack(side=TOP, pady=20)

#buttons
#Buttons
btn_login = Button(Form, text='Start', width=45, command=startApp)
btn_login.grid(row = 0, column = 0)

def main():
    root.mainloop()

if __name__ == '__main__':
    main() 