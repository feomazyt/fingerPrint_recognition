import cv2
import numpy as np
import os
import fingerprint_enhancer
from skimage.morphology import skeletonize
from skimage.util import invert

img1 = cv2.imread("Data/Magda1.bmp", cv2.IMREAD_GRAYSCALE)


# test_original = fingerprint_enhancer.enhance_Fingerprint(test_original)
# test_original[test_original >= 255] = 1
# test_original = skeletonize(test_original)
# test_original = np.asarray(invert(test_original), dtype=np.uint8)
# test_original[test_original >= 1] = 255
# cv2.imshow("Original", cv2.resize(test_original, None, fx=1, fy=1))
# cv2.waitKey()

def removedot(invertThin):
    temp0 = np.array(invertThin[:])
    temp0 = np.array(temp0)
    temp1 = temp0 / 255
    temp2 = np.array(temp1)
    temp3 = np.array(temp2)

    enhanced_img = np.array(temp0)
    filter0 = np.zeros((10, 10))
    W, H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize, j:j + filtersize]

            flag = 0
            if sum(filter0[:, 0]) == 0:
                flag += 1
            if sum(filter0[:, filtersize - 1]) == 0:
                flag += 1
            if sum(filter0[0, :]) == 0:
                flag += 1
            if sum(filter0[filtersize - 1, :]) == 0:
                flag += 1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize))

    return temp2


def get_descriptors(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = fingerprint_enhancer.enhance_Fingerprint(img)
    img = np.array(img, dtype=np.uint8)
    # Threshold
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Normalize to 0 and 1 range
    img[img == 255] = 1

    # Thinning
    skeleton = skeletonize(img)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = removedot(skeleton)
    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125
    # Extract keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
        for y in range(0, harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))
    # Define descriptor
    orb = cv2.ORB_create()
    # Compute descriptors
    _, des = orb.compute(img, keypoints)
    return (keypoints, des);


for file in [file for file in os.listdir("Data")]:
    img2 = cv2.imread("Data/" + file, cv2.IMREAD_GRAYSCALE)
    # fingerprint_database_image = fingerprint_enhancer.enhance_Fingerprint(fingerprint_database_image)
    # fingerprint_database_image[fingerprint_database_image >= 255] = 1
    # fingerprint_database_image = skeletonize(fingerprint_database_image)
    # fingerprint_database_image = np.asarray(invert(fingerprint_database_image), dtype=np.uint8)
    # fingerprint_database_image[fingerprint_database_image >= 1] = 255

    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = get_descriptors(img1)
    keypoints_2, descriptors_2 = get_descriptors(img2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(descriptors_1, descriptors_2), key=lambda match: match.distance)
    # Plot keypoints
    img4 = cv2.drawKeypoints(img1, keypoints_1, outImage=None)
    img5 = cv2.drawKeypoints(img2, keypoints_2, outImage=None)
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img4)
    # axarr[1].imshow(img5)
    # plt.show()
    # Plot matches
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
    # plt.imshow(img3)
    # plt.show()

    # Calculate score
    score = 0;
    for match in matches:
        score += match.distance
    score_threshold = 33
    if score / len(matches) < score_threshold:
        print("Fingerprint matches.")
    else:
        print("Fingerprint does not match.")

    # matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),
    #                                 dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    # match_points = []
    #
    # for p, q in matches:
    #     if p.distance < 0.1 * q.distance:
    #         match_points.append(p)
    #
    # keypoints = 0
    # if len(keypoints_1) <= len(keypoints_2):
    #     keypoints = len(keypoints_1)
    # else:
    #     keypoints = len(keypoints_2)
    # if (len(match_points) / keypoints) > 0.1:
    #     print("% match: ", len(match_points) / keypoints * 100)
    #     print("Figerprint ID: " + str(file))
    #     result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image,
    #                              keypoints_2, match_points, None)
    #     result = cv2.resize(result, None, fx=2.5, fy=2.5)
