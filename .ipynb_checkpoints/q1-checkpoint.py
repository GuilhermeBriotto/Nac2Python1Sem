import cv2
import numpy as np
from matplotlib import pyplot as plt
 
template = cv2.imread('Carta7OurosMaior.png',0)
#templateInv = cv2.imread('Carta7OurosMaiorInvertida.png',0)

template2 = cv2.imread('Carta7Ouros.png',0)
#template2Inv = cv2.imread('Carta7OurosInvertida.png',0)

face_w, face_h = template.shape[::-1]
#face_wInv, face_hInv = templateInv.shape[::-1]
face_w2, face_h2 = template2.shape[::-1]
#face_w2Inv, face_h2Inv = template2Inv.shape[::-1]


cap = cv2.VideoCapture("q1.mp4")

threshold = 0.8
### quanto mais próximo de 1 o valor do "threshold" mais branco, quanto mais próximo de 0 mais preto
threshold2 = 0.7

while True :
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Seu código aqui.
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ### Faz o Match Template
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    #resInv = cv2.matchTemplate(img_gray,templateInv,cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
    #res2Inv = cv2.matchTemplate(img_gray,template2Inv,cv2.TM_CCOEFF_NORMED)
    
    ### Prints para testar e definir o valor do "threshold"
    #print(res)
    #print(res2)
    
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #min_val2, max_val2, min_loc2, max_lo2c = cv2.minMaxLoc(res2)

    ### Desenhando o retângulo na carta maior
    location = np.where(res >= threshold)
    for pt in zip(*location[::-1]):
        #print(pt)
        cv2.rectangle(frame, pt, (pt[0] + face_w, pt[1] + face_h), (65, 255, 3), 4)
        ### Imprime "Carta Detectada" se caso desenhar o retângulo
        cv2.putText(frame, "Carta Detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    ### Desenhando o retângulo na carta menor
    location2 = np.where(res2 >= threshold2)
    for pt2 in zip(*location2[::-1]):
        #print(pt)
        cv2.rectangle(frame, pt2, (pt2[0] + face_w2, pt2[1] + face_h2), (65, 255, 3), 4)
        ### Imprime "Carta Detectada" se caso desenhar o retângulo
        cv2.putText(frame, "Carta Detectada", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    ### Mostra o frame(que vira um vídeo por causa da repetição), o título da tela é vídeo
    # Exibe resultado
    cv2.imshow('frame', frame)
    #cv2.imshow('image', res)
    #cv2.imshow('image', res2)
    
    # Wait for key 'ESC' to quit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()