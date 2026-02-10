import cv2

mask = cv2.imread("data/256/data 2/masks/00004.png", cv2.IMREAD_GRAYSCALE)

print(mask.shape)
