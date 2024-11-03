import cv2
import numpy as np

def generate_random_image(seed=1337, size=(10, 10)):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate a random 10x10 matrix with values between 0 and 255
    random_image = np.random.randint(0, 256, size, dtype=np.uint8)

    return random_image

def otsu(histogram) -> int:
    return np.argmax([
        np.sum(histogram[:k]) * (1 - np.sum(histogram[:k])) *
        (
             np.sum([i * pi for i, pi in enumerate(histogram[:k])])/np.sum(histogram[:k]) - 
             np.sum([(k + i) * pi for i, pi in enumerate(histogram[k:])]) / np.sum(histogram[k:])
        )**2 if np.sum(histogram[:k]) > 0 and np.sum(histogram[k:]) > 0 else 0
        for k in range(256)
    ])

np.random.seed(1337)
image = np.random.randint(0, 256, (14, 14), dtype=np.uint8)
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
histogram /= np.sum(histogram)
thresh = otsu(histogram) 
size = (400,400)
im1 = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
image[image < thresh] = 0
image[image >= thresh] = 255
im2 = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
concatenated_image = np.hstack((im1, im2))
cv2.imshow("", concatenated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('random_matrix_otsu.png', concatenated_image)
