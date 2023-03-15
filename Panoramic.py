import cv2
import numpy as np

#initilalize the SIFT detector
sift = cv2.SIFT_create()
path = ['image_1.jpg','image_2.jpg','image_3.jpg','image_4.jpg']
resized_image=[]
scale = 0.3
for images in path:
    img=cv2.imread(images)
    img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    resized_image.append(img_resized)
threshold = 0.5
#function to match points
def match_ponits_knn_flann(first_image,second_image):
    gray_img1= cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    gray_img2= cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray_img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray_img2, None)
    matching_features_1 = cv2.FlannBasedMatcher()
    features_matched_1 = matching_features_1.knnMatch(descriptors_1, descriptors_2, k=2)
    best_matched_keypoints = []
    for i,j in features_matched_1:
        if i.distance < threshold*j.distance:
            best_matched_keypoints.append(i)     
    image_match = cv2.drawMatches(gray_img1, keypoints_1, gray_img2, keypoints_2, best_matched_keypoints, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return image_match

#function to combine images
def panaroma(first_image,second_image):
    #converting to gray scale
    gray_img1= cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    gray_img2= cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
    #getting keypoints and descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray_img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray_img2, None)
    #FlannMatcher
    matching_features_1 = cv2.FlannBasedMatcher()
    features_matched_1 = matching_features_1.knnMatch(descriptors_1, descriptors_2, k=2)
    best_matched_keypoints = []
    for i,j in features_matched_1:
        if i.distance < threshold*j.distance:
            best_matched_keypoints.append(i) 
    src_pt1 = np.float32([keypoints_1[m.queryIdx].pt for m in best_matched_keypoints ]).reshape(-1,1,2)
    dest_pt1 = np.float32([keypoints_2[m.trainIdx].pt for m in best_matched_keypoints ]).reshape(-1,1,2)
    threshold_RANSAC = 10
    iterative = 1000
    best_hmatrix = None
    min_no_liner = 0
    #iterating the loop for 
    for i in range(iterative):
        index = np.random.choice(len(src_pt1), 4, replace= False)
        samp1 = src_pt1[index]
        samp2 = dest_pt1[index]
        #declaring amatrix list
        Mat_List = []
        for i in range(len(samp2)):
            x_src, y_src = samp2[i][0][0], samp2[i][0][1]
            x_h, y_h = samp1[i][0][0], samp1[i][0][1]
            Mat_List.append(np.array([
            [x_src, y_src, 1, 0, 0, 0, -x_h*x_src, -x_h*y_src, -x_h],
            [0, 0, 0, x_src, y_src, 1, -y_h*x_src, -y_h*y_src, -y_h]
        ]))
            matrix_A = np.empty([0, Mat_List[0].shape[1]])
        for i in Mat_List:
         matrix_A = np.append(matrix_A, i, axis=0)
        #finding the eigen values
        eigen1, eigen2 = np.linalg.eig(matrix_A.T @ matrix_A)
        homogr = eigen2[:, np.argmin(eigen1)]
        homograpghy_matrix = homogr.reshape((3, 3))
        source = np.concatenate(
        (dest_pt1, np.ones((len(dest_pt1), 1, 1), dtype=np.float32)), axis=2)
        homography_points = np.matmul(source, homograpghy_matrix.T)
        points = homography_points[:, :, :2] / homography_points[:, :, 2:]
        d = np.linalg.norm(src_pt1 - points, axis = 2)
        inliner = np.sum(d<threshold_RANSAC)

        if inliner > min_no_liner:
            min_no_liner = inliner
            best_hmatrix = homograpghy_matrix  
    stitch_image = cv2.warpPerspective(second_image, best_hmatrix, ((first_image.shape[1] + second_image.shape[1]), first_image.shape[0]))
    stitch_image[0:first_image.shape[0], 0:first_image.shape[1]] = first_image
    return stitch_image

# To match and visualise the features between consecutive images:
for i in range(len(resized_image) - 1):
    # Match the features and visualize them in the first and second image
    stitched = match_ponits_knn_flann(resized_image[i], resized_image[i+1])
    cv2.imshow('Features match_ponits_knn_flann', stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#To combine these four images together:
panaroma_1 = panaroma(resized_image[2],resized_image[3])
panaroma_2 = panaroma(resized_image[1], panaroma_1)
Final_panaroma = panaroma(resized_image[0], panaroma_2)
resize_panaroma = cv2.resize(Final_panaroma, None, fx=scale, fy=scale)
cv2.imshow('Final panaroma', resize_panaroma)
cv2.imwrite('panaroma.jpg',resize_panaroma)
cv2.waitKey(0)
cv2.destroyAllWindows()