import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
class Graph:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.graph = np.zeros((height, width, 3), np.uint8)
    def update_frame(self, value):
        if value < 0:
            value = 0
        elif value >= self.height:
            value = self.height - 1
        new_graph = np.zeros((self.height, self.width, 3), np.uint8)
        new_graph[:,:-1,:] = self.graph[:,1:,:]
        new_graph[self.height - value:,-1,:] = 255
        self.graph = new_graph
    def get_graph(self):
        return self.graph

img = cv2.imread('lena.png')
cap = cv2.VideoCapture(0)
cap.set(3,620)
cap.set(4,480)
graph = Graph(100, 60)
prev_frame = np.zeros((480, 640), np.uint8)
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
c = []
for x in range(0,10):
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img , confThreshold=0.5)
    print(classIds,bbox)
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img, str(round(confidence*100,2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    print(confs)
    c.append(confs[0])
    pred_scores = confs
    print( pred_scores )
    if(pred_scores>=0.5).any():
        print("positive")
    else:
        (pred_scores<0.5).any()
        print("negative")
print(c)

df = pd.DataFrame(c)
df.columns = ['Confidence values']
df.to_csv('data1.csv', header=True,index=True)
a = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for i,j in zip(c,a):
    plt.plot(a,c)
    plt.xlabel('x - axis')
        # naming the y axis
    plt.ylabel('y - axis')
        # giving a title to my graph
    plt.title('Confidence Graph')
        # function to show the plot
plt.show()
while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), None)
    diff = cv2.absdiff(prev_frame, gray)
    difference = np.sum(diff)
    prev_frame = gray
    graph.update_frame(int(difference/42111))
    roi = frame[-70:-10, -110:-10,:]
    roi[:] = graph.get_graph()
    cv2.putText(frame, "drone movement", (20, 430), cv2.FONT_HERSHEY_PLAIN, 1.8, (200, 200, 200), 2)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break






