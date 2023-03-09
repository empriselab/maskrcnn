import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../data/video/00000.png')

# will fail if setup is bad
#cv2.imshow('test', img)
#cv2.waitKey(0)

plt.imshow(img)
plt.show()
