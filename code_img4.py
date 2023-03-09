import cv2
import numpy as np
import pytesseract as tesseract

img = cv2.imread("imagens\img_7.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 50, 150)
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)
mask = np.zeros_like(img)
cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
out = np.zeros_like(img)
out[mask == 255] = img[mask == 255]

# Obter as coordenadas do contorno da placa
y, x, _ = np.where(mask == 255)
top_left = (np.min(x), np.min(y))
bottom_right = (np.max(x), np.max(y))

# Recortar a região da imagem correspondente à placa
out_cropped = out[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1]

# Redimensionar a imagem para um tamanho fixo (por exemplo, 400x100 pixels)
out_resized = cv2.resize(out_cropped, (400, 100))

img_filtrada =  cv2.medianBlur(out_resized, 3)
kernel = np.ones((3,3),np.float32)/9
img_filtrada = cv2.filter2D(img_filtrada,-1,kernel)
# Salvar a imagem resultante em um arquivo
# cv2.imwrite("placa_sem_fundo.png", out_resized)
cv2.imshow("placa", img_filtrada)


# apontando tesseract pro executavel
tesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\Tesseract.exe"

result = tesseract.image_to_string(img_filtrada)

print(result)

cv2.waitKey(0)
cv2.destroyAllWindows()