import cv2
import numpy as np

# Ouverture du flux video
cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

ret, frame1 = cap.read()
yuv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)

hist = cv2.calcHist([yuv1], [1, 2], None, [256, 256], [0, 256, 0, 256])
hist = cv2.GaussianBlur(hist, (5, 5), 0)
hist_log = np.log(hist + 1)
hist_norm1 = cv2.normalize(hist_log, None, 0, 255, cv2.NORM_MINMAX)
hist_image = cv2.applyColorMap(
    hist_norm1.astype(np.uint8), cv2.COLORMAP_JET)

cv2.imshow('Histogramme 2D', hist_image)
cv2.imshow('Image originale', frame1)


ret, frame2 = cap.read()
yuv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)


while True:

    ret, frame2 = cap.read()
    if not ret:
        break
    yuv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)

    hist = cv2.calcHist([yuv2], [1, 2], None, [256, 256], [0, 256, 0, 256])
    hist = cv2.GaussianBlur(hist, (5, 5), 0)
    hist_log = np.log(hist + 1)
    hist_norm2 = cv2.normalize(hist_log, None, 0, 255, cv2.NORM_MINMAX)
    hist_image = cv2.applyColorMap(
        hist_norm2.astype(np.uint8), cv2.COLORMAP_JET)

    diff = cv2.absdiff(yuv1[:, :, 1:], yuv2[:, :, 1:])
    diff = diff[:, :, 1]+diff[:, :, 0]
    diff = diff.reshape(1, -1)[0]

    # _, modifs = cv2.threshold(diff, 55, 1, cv2.THRESH_BINARY)

    # print(sum(modifs))

    frame1 = frame2
    yuv1 = yuv2
    hist_norm1 = hist_norm2

    cv2.imshow('Histogramme 2D', hist_image)
    #cv2.imshow('Image originale', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ferme la fenêtre d'affichage et libère le flux vidéo
cap.release()
cv2.destroyAllWindows()