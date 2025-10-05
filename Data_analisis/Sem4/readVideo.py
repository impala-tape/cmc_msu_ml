import cv2
import matplotlib.pyplot as plt

source = 'race_car_small.mp4'

cap = cv2.VideoCapture(source)

if(not cap.isOpened()):
    print('Error in opening video file')
    exit(1)

# ret, fram = cap.read()

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (640,  480))
count = 0
while(cap.isOpened() and count < 100):
    ret, frame = cap.read()
    count += 1

    if ret == True:
        #plt.imshow(frame[:,:,::-1])
        #grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flip = cv2.flip(frame, 1)
        out.write(flip)
        #if cv2.waitKey(10) == ord('q'):
        #    break
 
cap.release()
cv2.destroyAllWindows()
out.release()