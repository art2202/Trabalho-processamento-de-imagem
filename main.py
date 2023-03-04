import cv2 as opencv
import pytesseract as tesseract




img = opencv.imread("imagens/img.png")

# apontando tesseract pro executavel
tesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\Tesseract.exe"

result = tesseract.image_to_string(img)

print(result)
