import numpy as np
import cv2

from matplotlib import pyplot as plt

import sys
if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze) match_method (= bf ou flann)")
  sys.exit(2)


transormation = input('\nTransformation:\n0 : pas de transformation\n1 : resize\n2 : resize2\n3 : translation\n4 : rotation\n5 : transformation affine\n6 : Perspective\nChoix : ')
#Lecture de la paire d'images
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
img2 = cv2.imread('../Image_Pairs/torb_small2.png')
print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
print("Type de l'image 2 :",img2.dtype)

rows,cols,ch = img1.shape
#TRANSFORMATION 1 - resize
img1_1 = cv2.resize(img1,(3*cols, 2*rows), interpolation = cv2.INTER_CUBIC)

#TRANSFORMATION 2 - resize
img1_2 = cv2.resize(img1,(2*cols, 3*rows), interpolation = cv2.INTER_CUBIC)

#TRANSFORMATION 3 - translation
M = np.float32([[1,0,100],[0,1,50]])
img1_3 = cv2.warpAffine(img1,M,(cols,rows))

#TRANSFORMATION 4 - rotation
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
img1_4 = cv2.warpAffine(img1,M,(cols,rows))

#TRANSFORMATION 5 - transformation affine
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
img1_5 = cv2.warpAffine(img1,M,(cols,rows))

#TRANSFORMATION 6 - perspective
pts1 = np.float32([[156,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
img1_6 = cv2.warpPerspective(img1,M,(300,300))


#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 500,#Par défaut : 500
                       scaleFactor = 1.2,#Par défaut : 1.2
                       nlevels = 8)#Par défaut : 8
  kp2 = cv2.ORB_create(nfeatures=500,
                        scaleFactor = 1.2,
                        nlevels = 8)
  print("Détecteur : ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
    		        threshold = 0.001,#Par défaut : 0.001
  		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")


gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#pas de transformation :
if transormation=="0":
  #Conversion en niveau de gris
  gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
elif transormation=="1":
  plt.subplot(121)
  plt.imshow(img1,cmap = 'gray')
  plt.title('Image originale')

  plt.subplot(122)
  plt.imshow(img1_1,cmap = 'gray')
  plt.title('Image transformée')

  plt.show()
  print("\n transfo 1!\n")
  gray2 =  cv2.cvtColor(img1_1,cv2.COLOR_BGR2GRAY)

elif transormation=="2":
  plt.subplot(121)
  plt.imshow(img1,cmap = 'gray')
  plt.title('Image originale')

  plt.subplot(122)
  plt.imshow(img1_2,cmap = 'gray')
  plt.title('Image transformée')

  plt.show()
  print("\n transfo 2!\n")
  gray2 =  cv2.cvtColor(img1_2,cv2.COLOR_BGR2GRAY)

elif transormation=="3":
  plt.subplot(121)
  plt.imshow(img1,cmap = 'gray')
  plt.title('Image originale')

  plt.subplot(122)
  plt.imshow(img1_3,cmap = 'gray')
  plt.title('Image transformée')

  plt.show()
  print("\n transfo 3!\n")
  gray2 =  cv2.cvtColor(img1_3,cv2.COLOR_BGR2GRAY)

elif transormation=="4":
  plt.subplot(121)
  plt.imshow(img1,cmap = 'gray')
  plt.title('Image originale')

  plt.subplot(122)
  plt.imshow(img1_4,cmap = 'gray')
  plt.title('Image transformée')

  plt.show()
  print("\n transfo 4!\n")
  gray2 =  cv2.cvtColor(img1_4,cv2.COLOR_BGR2GRAY)

elif transormation=="5":
  plt.subplot(121)
  plt.imshow(img1,cmap = 'gray')
  plt.title('Image originale')

  plt.subplot(122)
  plt.imshow(img1_5,cmap = 'gray')
  plt.title('Image transformée')

  plt.show()
  print("\n transfo 5!\n")
  gray2 =  cv2.cvtColor(img1_5,cv2.COLOR_BGR2GRAY)

elif transormation=="6":
  plt.subplot(121)
  plt.imshow(img1,cmap = 'gray')
  plt.title('Image originale')

  plt.subplot(122)
  plt.imshow(img1_6,cmap = 'gray')
  plt.title('Image transformée')

  plt.show()
  print("\n transfo 6!\n")
  gray2 =  cv2.cvtColor(img1_6,cv2.COLOR_BGR2GRAY)


#Détection et description des keypoints
pts1, desc1 = kp1.detectAndCompute(gray1,None)
pts2, desc2 = kp2.detectAndCompute(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")
# Calcul de l'appariement
t1 = cv2.getTickCount()
if detector == 1:
  #Distance de Hamming pour descripteur BRIEF (ORB)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
else:
  #Distance L2 pour descripteur M-SURF (KAZE)
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# Extraction de la liste des 2-plus-proches-voisins
matches = bf.knnMatch(desc1,desc2, k=2)
# Application du ratio test
good = []
for m,n in matches:
  if m.distance < 0.7*n.distance:
    good.append([m])
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de l'appariement :",time,"s")

# Affichage des appariements qui respectent le ratio test
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 0)
img3 = cv2.drawMatchesKnn(gray1,pts1,gray2,pts2,good,None,**draw_params)

Nb_ok = len(good)
plt.imshow(img3),plt.title('%i appariements OK'%Nb_ok)
plt.show()


