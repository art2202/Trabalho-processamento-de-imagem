import re
import cv2
import numpy as np
import pytesseract as tesseract

tesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\Tesseract.exe"

file = cv2.imread("imagens\img_10.png")


def canny(img_gray):
    edged = cv2.Canny(img_gray, 0, 0)
    cv2.imshow("Canny", edged)
    return edged


def cvt_color(img, cv2_color):
    gray = cv2.cvtColor(img, cv2_color)
    cv2.imshow("cvtColor", gray)
    return gray


def gaussian(img_gray):
    gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    cv2.imshow("GaussianBlur", gray)
    return gray


def function_contours(img, edged):
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]
    return out, mask


def out_cropped(out, mask):
    # Obter as coordenadas do contorno da placa
    y, x, _ = np.where(mask == 255)
    top_left = (np.min(x), np.min(y))
    bottom_right = (np.max(x), np.max(y))

    # Recortar a região da imagem correspondente à placa
    out_cropped = out[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
    cv2.imshow("out_cropped", out_cropped)
    return out_cropped


def filter_2d(image):
    kernel = np.ones((3, 3), np.float32) / 9
    img = cv2.filter2D(image, -1, kernel)
    cv2.imshow("filter2D", img)
    return img


def median_blur(image):
    img_filtered = cv2.medianBlur(image, 3)
    cv2.imshow("medianBlur", img_filtered)
    return img_filtered


def thresholding(image):
    image_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow("threshold", image_threshold)
    return image_threshold


def erode(image):
    kernel = np.ones((2, 2), np.uint8)
    img_filtered = cv2.erode(image, kernel, iterations=1)
    cv2.imshow("erode", img_filtered)
    return img_filtered


def dilate(image):
    kernel = np.ones((2, 2), np.uint8)
    img_filtered = cv2.dilate(image, kernel, iterations=1)
    cv2.imshow("dilate", img_filtered)
    return img_filtered


def print_result(img_result):
    config = ('-l eng --oem 1 --psm 3')
    result = tesseract.image_to_string(img_result, config=config)
    print(result)


img_grey = cvt_color(file, cv2.COLOR_BGR2GRAY)
img_gauss = median_blur(img_grey)
img_thresh = thresholding(img_gauss)
img_dilate = dilate(img_gauss)
img_erode = erode(img_dilate)
img_canny = canny(img_erode)
out, mask = function_contours(file, img_canny)
img_bolada = out_cropped(out, mask)
result = tesseract.image_to_string(img_bolada)
texto_sem_especiais = re.sub(r'[^\w\s]|_+$', '', result)
# texto_sem_caracteres_especiais = re.sub('[^A-Za-z0-9\s\.]+', '', result)

print(texto_sem_especiais)
cv2.waitKey(0)
cv2.destroyAllWindows()
