import cv2
import sys
from pathlib import Path

path = "Forest_Gump_mountains.mp4"
# if len(sys.argv) > 1:
#     path = sys.argv[1]

src = cv2.VideoCapture(path)

winName = "Face detection"
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)


# Загружаем модель
net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\Mishele Dolmin\Documents\CODE\cmc_msu_ml\Data_analisis\Sem7\deploy.prototxt", 
    r"C:\Users\Mishele Dolmin\Documents\CODE\cmc_msu_ml\Data_analisis\Sem7\res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

inputWidth, inputHeight = 300, 300
mean = [104, 117, 123]
confidenceThreshold = 0.1

while cv2.waitKey(40) != 27 :
    ok, frame = src.read()
    if not ok:
        break

    frameHeight, frameWidth = frame.shape[0], frame.shape[1]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inputWidth, inputHeight), mean, swapRB = False, crop = False )
    net.setInput(blob)

    detections = net.forward()
    
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidenceThreshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)            

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "Conf : %.5f" % conf

            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    t, _ = net.getPerfProfile()
    label2 = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    cv2.imshow(winName, frame)

src.release()
cv2.destroyWindow(winName)