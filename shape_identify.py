import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpimg



img=cv2.imread('/home/gaurav/TrafficDataset/proj1.png')
image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, contours, _ =cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Number of shapes {0}",format(len(contours)))

for cnt in contours:
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    img = cv2.drawContours(img,[box],0,(0,255,0),3)

plt.figure("Example1")
plt.imshow(img)
plt.title("Binary Contours in an image")
plt.show()
lane1=10*len(contours)

#################################################

img1=cv2.imread('/home/gaurav/TrafficDataset/proj2.png')
image=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
_, contours, _ =cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Number of shapes {0}",format(len(contours)))

for cnt in contours:
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    img1 = cv2.drawContours(img1,[box],0,(0,255,0),3)

plt.figure("Example2")
plt.imshow(img1)
plt.title("Binary Contours in an image")
plt.show()
lane2=10*len(contours)

############################################################

img2=cv2.imread('/home/gaurav/TrafficDataset/proj3.png')
image=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, contours, _ =cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Number of shapes {0}",format(len(contours)))

for cnt in contours:
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    img2 = cv2.drawContours(img2,[box],0,(0,255,0),3)

plt.figure("Example3")
plt.imshow(img2)
plt.title("Binary Contours in an image")
plt.show()
lane3=10*len(contours)

###############################################################

img2=cv2.imread('/home/gaurav/TrafficDataset/cars1.png')
image=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, contours, _ =cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Number of shapes {0}",format(len(contours)))

for cnt in contours:
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    img2 = cv2.drawContours(img2,[box],0,(0,255,0),3)

plt.figure("Example3")
plt.imshow(img2)    
plt.title("Binary Contours in an image")
plt.show()
lane4=10*len(contours)
############################################


print("\n\nThe Traffic in lanes is as follows :\n")

print("LANE 1: ",lane1)
print("LANE 2: ",lane2)
print("LANE 3: ",lane3)
print("LANE 4: ",lane4)
lanes=[lane1,lane2,lane3,lane4]
max=lanes[0]
lane_no="Lane4"
for i in range(4):
    if(max<lanes[i]):
        max=lanes[i]

print("Maximum Traffic is ",max)
