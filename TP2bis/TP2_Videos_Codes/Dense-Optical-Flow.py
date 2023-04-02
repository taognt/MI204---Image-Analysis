import cv2
import numpy as np
from matplotlib import pyplot as plt


# True si on veut afficher la similarité entre deux images consecutives en temps reel dans le terminal
#La moyenne est egalement affichée a chaque appui sur "p" sur le clavier
Test_similarite= False

#Ouverture du flux video
cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
prvs_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV) # Passage en YUV

hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
next_yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)

#nombre de fois que l'on a comparé les histogrammes des flots
nbp  = 0
#moyenne de la similarité entre deux flots consécutifs
sum_similarite = 0
moy_similarite = 0


while(ret):
    index += 1
    
    #Entre deux images consécutives :
    flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)	
    
    #recuperation des composantes Vx et Vy du flot :
    Vx, Vy = cv2.split(flow)

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 

    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2,bgr))
    cv2.imshow('Image et Champ de vitesses (Farneback)',result)

    #Q1 : Histogramme 2D du codage YUV des images couleur
    Yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
    hist = cv2.calcHist([Yuv], [1,2], None, [256,256], [0, 255,0, 255])
    hist = cv2.GaussianBlur(hist, (5,5), 0)
    hist_log = np.log(hist+1)
    hist_norm2 = cv2.normalize(hist_log, None, 0, 255, cv2.NORM_MINMAX)
    hist_image = cv2.applyColorMap(hist_norm2.astype(np.uint8), cv2.COLORMAP_JET)

    #Q4 : Histogramme 2D correspondant à la probabilité jointe des composantes (Vx, Vy) du flot optique
    hist_Flot = cv2.calcHist([Vx, Vy], [0,1], None, [256,256], [0,255,0,255])
    hist_Flot_log = np.log(hist_Flot+1)
    hist_Flot_norm = cv2.normalize(hist_Flot_log, None, 0, 255, cv2.NORM_MINMAX)
    hist_image_flot = cv2.applyColorMap(hist_Flot_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.normalize(hist_Flot, hist_Flot, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    #Q5 : comparaison de deux histogrammes consécutifs | similarite = 1 => memes flots

    if index == 2:
        prev_hist_flot = hist_Flot.copy()
    
    similarite_flot = cv2.compareHist(hist_Flot, prev_hist_flot, cv2.HISTCMP_CORREL)
    nbp += 1
    sum_similarite += similarite_flot
    moy_similarite = sum_similarite/nbp

    if Test_similarite:
        print("similarité entre les histogrammes des flots : ", similarite_flot)

    prev_hist_flot = hist_Flot.copy()

    #cv2.imshow('Vx', Vx)
    #cv2.imshow('Vy', Vy)
    cv2.imshow('Histogramme 2D',hist_image)
    cv2.imshow('Histogramme 2D du flot', hist_image_flot)

    #capture de l'image changement de plan :
    if(similarite_flot < 0.89):
        cv2.imwrite('./images/Frame_%04d.png'%index,frame2)
        #cv2.imwrite('OF_hsv_%04d.png'%index,bgr)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)

    #afficher la similarité et la moyenne de la similarité entre deux histogrammes de flots consécutifs :
    elif k == ord('p'):
        moy_similarite = sum_similarite/nbp
        print("\n-------------\n")
        print("Moyenne : ", moy_similarite)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

cap.release()
cv2.destroyAllWindows()
