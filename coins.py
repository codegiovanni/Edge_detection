import numpy as np
import cv2


def nothing(x):
    pass


cv2.namedWindow("Trackbar")
cv2.createTrackbar("Threshold1", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("Threshold2", "Trackbar", 0, 255, nothing)

while True:
    img = cv2.imread("input/coins.jpg")
    img_original = img.copy()

    thresh1 = cv2.getTrackbarPos("Threshold1", "Trackbar")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Trackbar")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edged = cv2.Canny(blurred, thresh1, thresh2, 3)
    dilated = cv2.dilate(edged, (1, 1), iterations=2)

    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Modifies image shape for stacking
    gray = np.stack((gray,) * 3, axis=-1)
    blurred = np.stack((blurred,) * 3, axis=-1)
    edged = np.stack((edged,) * 3, axis=-1)
    dilated = np.stack((dilated,) * 3, axis=-1)

    images = [gray, blurred, edged, dilated]
    win_names = ['Gray', 'Blurred', 'Edged', 'Dilated']

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Horizontal window stacking
    img_stack = np.hstack(images)
    for index, name in enumerate(win_names):
        image = cv2.putText(img_stack, f'{index + 1}. {name}', (5 + img.shape[1] * index, 30),
                            font, 1, (205, 0, 255), 2, cv2.LINE_AA)

    # Vertical window stacking
    # img_stack = np.vstack(images)
    # for index, name in enumerate(win_names):
    #     image = cv2.putText(img_stack, f'{index + 1}. {name}', (5, 30 + img.shape[0] * index),
    #                         font, 1, (205, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Image Processing", img_stack)

    cv2.drawContours(img_original, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Output", img_original)

    print("Coins in the image:", len(contours))

    cv2.waitKey(300)
