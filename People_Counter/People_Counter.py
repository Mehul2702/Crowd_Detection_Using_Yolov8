from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap=cv2.VideoCapture('../videos/people.mp4') #for video

model=YOLO('../yolo-weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask=cv2.imread("mask.png")

# Tracker
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

# line
limitsup=[103,161,296,161]
limitsdown=[560,489,760,487]

totalCountsup=[]
totalCountsdown=[]

while(True):
    success , img=cap.read()
    imgRegion=cv2.bitwise_and(img,mask)
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:

            # Bounding Box

            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)

            # confidence

            confidence=box.conf[0]
            confidence=math.ceil((box.conf[0]*100))/100;
            print(confidence)

            cls=int(box.cls[0])
            currentClass=classNames[cls]
            if(currentClass=="person" and confidence>0.3):
                # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9,rt=5)
                # cvzone.putTextRect(img,f'{classNames[cls]} {confidence}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                currentArray=np.array([x1,y1,x2,y2,confidence])
                detections=np.vstack((detections,currentArray))


    resultsTracker=tracker.update(detections)
    cv2.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,0,255),4)
    cv2.line(img,(limitsdown[0],limitsdown[1]),(limitsdown[2],limitsdown[3]),(0,0,255),4)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 ,id= int(x1), int(y1), int(x2), int(y2),int(id)
        print(result)
        cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=1,
                           offset=3)
        #     finding center
        cx,cy=x1+(x2-x1)//2,y1+(y2-y1)//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limitsup[0]-10<cx<limitsup[2]+10 and limitsup[1]-50<cy<limitsup[1]+50:
            if totalCountsup.count(id)==0:
                totalCountsup.append(id)
        if limitsdown[0] - 10 < cx < limitsdown[2] + 10 and limitsdown[1] - 50 < cy < limitsdown[1] + 50:
            if totalCountsdown.count(id)==0:
                totalCountsdown.append(id)

    cvzone.putTextRect(img, f'Counts Up : {len(totalCountsup)} Counts Down : {len(totalCountsdown)}', (50, 50), scale=2, thickness=1,
                       offset=3)


    cv2.imshow("Image",img)
    # cv2.imshow("Imageregion", imgRegion)
    cv2.waitKey(1)
