import functions
from yolopy import yolo
import speech


import cv2
import os
import detect
import datetime


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "dfkey.json"

labelsPath = "D:/PycharmProjects/intern proj/yolo/coco.names"
weightsPath = "D:/PycharmProjects/intern proj/yolo/yolov3.weights"
configPath = "D:/PycharmProjects/intern proj/yolo/yolov3.cfg"


args = {"threshold":0.3, "confidence":0.5}
project_id = "jyothy-jjdc"
#project_id = "blindbot-286ed"

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

engine = speech.speech_to_text()

model = yolo(labelsPath, weightsPath, configPath)

listening = False
intent = None
while True:

    if not listening:
        resp = engine.recognize_speech_from_mic()
        print(resp)
        if(resp != None):
            intent, text = detect.detect_intent_texts(project_id, 0, [resp], 'en')
        if(intent == 'Jyoti' and resp!=None):
            listening = True

    else:
        engine.text_speech("What can I help you with?")
        intent = ''
        engine.text_speech("Listening")
        resp = engine.recognize_speech_from_mic()
        engine.text_speech("Processing")
        if(resp!=None):
            print(resp)
            intent, text = detect.detect_intent_texts(project_id, 0, [resp], 'en')
        if intent == 'Describe':
            detect.describeScene(cam, model, engine)
        elif intent == 'endconvo':
            print(text)
            listening = False
            engine.text_speech(text)
        elif intent == 'Brightness':
            engine.text_speech("It is {} outside".format((functions.getBrightness(cam))[0]))
        elif intent == "FillForm":
            detect.detect_form(cam, engine)
        elif intent == "Read":
            print("read")
            detect.detect_text(cam, engine)
        elif intent == "Time":
            currentDT = datetime.datetime.now()
            engine.text_speech("The time is {} hours and {} minutes".format(currentDT.hour, currentDT.minute))
        elif resp != 'None':
            engine.text_speech(text)

    cam.release()
    cv2.waitKey(1)

