import dlib
from imutils import face_utils
import cv2
import numpy as np
from scipy.spatial import distance as dist
import threading
import pygame
import imutils as imutils

def start_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("z.ogg")
    pygame.mixer.music.play()

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36,48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
	
def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
   
	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
    return ear
	
camera = cv2.VideoCapture(0)

def Zoom(cv2Object, zoomSize):
    # Resizes the image/video frame to the specified amount of "zoomSize".
    # A zoomSize of "2", for example, will double the canvas size
    cv2Object = imutils.resize(cv2Object, width=(zoomSize * cv2Object.shape[1]))
    # center is simply half of the height & width (y/2,x/2)
    center = (cv2Object.shape[0]/2,cv2Object.shape[1]/2)
    # cropScale represents the top left corner of the cropped frame (y/x)
    cropScale = (center[0]/zoomSize, center[1]/zoomSize)
    # The image/video frame is cropped to the center with a size of the original picture
    # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
    # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
    cv2Object = cv2Object[int(cropScale[0]):(int(center[0]) + int(cropScale[0])), int(cropScale[1]):(int(center[1]) + int(cropScale[1]))]
    return cv2Object

	
predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total=0
alarm=False
while True:
    ret, frame = camera.read()
    
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=300)
    
# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    flag = 0 
	
    if len(dets) == 0:
        cv2.putText(frame, "Keep your eyes on the road!", (190,250),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 2)

    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            leftEye= shape[lStart:lEnd]
            rightEye= shape[rStart:rEnd]
            leftEAR= eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            roundEar = "{0:.2f}".format(ear)
            leftEyeHull = cv2.convexHull(leftEye)
	       
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), -1)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), -1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
            if ear == 0:
                cv2.putText(frame, "Please keep your eyes on the road!", (100,250),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            if ear>.25:
                print (ear)
                total=0
                alarm=False
                cv2.putText(frame, "Eyes Open ", (180, 200),cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, roundEar, (180, 230),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
            else:
                total+=1
                print("Eyes closed for: ", total, " iterations")
                if total>15:
                    cv2.putText(frame, "****************ALERT!****************", (100, 150),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "****************ALERT!****************", (100, 250),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (100,350),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, str(total-16), (200, 300),cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                    if not alarm:
                        alarm=True
                        d=threading.Thread(target=start_sound)
                        d.setDaemon(True)
                        d.start()

                        print ("Eyes have been closed for ", total, " frames show alert!")
                        cv2.putText(frame, "****************ALERT!****************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
                cv2.putText(frame, "Eyes Closed".format(total), (180, 200),cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, roundEar, (180, 230),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 1, (78, 248, 236), -1)
    
    cv2.imshow("image", Zoom(frame,2))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
