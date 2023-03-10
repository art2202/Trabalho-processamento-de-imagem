import cv2
import numpy as np
import pytesseract as tesseract
from pytesseract import Output

tesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\Tesseract.exe"

'''img = cv2.imread("imagens\img_8.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("teste", gray)
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
# out_resized = cv2.resize(out_cropped, (500, 300))

img_filtrada = cv2.medianBlur(out_cropped, 3)
kernel = np.ones((3,3),np.float32)/9
img_filtrada = cv2.filter2D(img_filtrada,-1,kernel)
# Salvar a imagem resultante em um arquivo
# cv2.imwrite("placa_sem_fundo.png", out_resized)
cv2.imshow("placa", img_filtrada)


# apontando tesseract pro executavel


config = ('-l eng --oem 1 --psm 3')
result = tesseract.image_to_string(img_filtrada, config=config)

# cleanResult = re.sub('[^A-Za-z0-9" "]+', ' ', result)


print(result)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''



image = cv2.imread("imagens\img_12.png")
# image = cv2.medianBlur(image, 3)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = tesseract.image_to_data(rgb, output_type=Output.DICT)

# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    conf = int(results["conf"][i])

# filter out weak confidence text localizations
    if conf > 50:
        # display the confidence and text to our terminal
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print("")
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw a bounding box around the text along
        # with the text itself
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)