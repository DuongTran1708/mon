import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
	IMAGE_H = 1080
	IMAGE_W = 1920

	src  = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	dst  = np.float32([[569, IMAGE_H], [1000, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	M    = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
	Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation

	img = cv2.imread(
		'/media/sugarubuntu/DataSKKU2/2_Dataset/TSS_Korea/rain/IMG_1954_non_sound.png')  # Read the test img
	img = img[450:(450 + IMAGE_H), 0:IMAGE_W]  # Apply np slicing for ROI crop
	warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))  # Image warping
	plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  # Show results
	plt.show()


if __name__ == "__main__":
	main()
