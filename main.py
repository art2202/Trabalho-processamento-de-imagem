import cv2 as opencv
import pytesseract as tesseract

# C:\Program Files\Tesseract-OCR
# pip install --force-reinstall --no-cache -U opencv-python==4.5.5.62 // instalar opencv com esse comando
# pip install pytesseract
# https://www.youtube.com/watch?v=GMqFZ7f0dy4&ab_channel=RonanVico // video tutorial disso tudo
# https://github.com/UB-Mannheim/tesseract/wiki // link do tesseract pra baixar o instalador


img = opencv.imread("imagens/img_1.png")

# apontando tesseract pro executavel
tesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\Tesseract.exe"

result = tesseract.image_to_string(img)

print(result)
