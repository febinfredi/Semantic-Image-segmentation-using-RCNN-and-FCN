import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import tkinter as tk
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# function for resizing image 
def resize_image(img, area=0.0, window_h=0, window_w=0):
    h, w = img.shape[:2]
    root = tk.Tk()
    screen_h = root.winfo_screenheight()
    screen_w = root.winfo_screenwidth()

    if area != 0.0:
        vector = math.sqrt(area)
        window_h = screen_h * vector
        window_w = screen_w * vector

    if h > window_h or w > window_w:
        if h / window_h >= w / window_w:
            multiplier = window_h / h
        else:
            multiplier = window_w / w
        img = cv2.resize(img, (0, 0), fx=multiplier, fy=multiplier)

    return img 


nearest_circle = []
curr_dist = 0  
   
img1 = cv2.imread(r"C:\Users\hp\Downloads\base_1_resize.jpeg")  # query Image
img2 = cv2.imread(r"C:\Users\hp\Downloads\mag5_resize.jpeg") # train Image

img_name = "mag_1.jpeg"
img2=resize_image(img2,area=0.8)
img_final = img2
img1=cv2.GaussianBlur(img1,(11,11),cv2.BORDER_DEFAULT)
img2=cv2.GaussianBlur(img2,(11,11),cv2.BORDER_DEFAULT)

# Create an ORB object
orb = cv2.ORB_create()

# find the keypoints and descriptors using ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
good_matches = matches[:50]

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)

# Show all the matches
plt.imshow(img3),plt.show()

#  Source points (from query image)
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
#  Destination points (from train image)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
#  Homography
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts,M)
dst += (w, 0)  # adding offset

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               # matchesMask = matchesMask, # draw only inliers
               flags = 2)
t1=np.reshape(dst_pts,(len(dst_pts),2))

# find meadian of matched keypoint
point=np.median(t1,axis=0)
k,l=int(point[0]),int(point[1])

# Draw keypoints matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

# Draw bounding box on target image and show along with train image 
img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
img3=cv2.resize(img3,(854,480))

# Print location of median
print(f"median: {k}, {l}") 

# Applying canny
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
center=(0,0)
radius=0
gray=cv2.GaussianBlur(gray,(25,25),cv2.BORDER_DEFAULT)
gray=cv2.Canny(gray,20,30) #20,30
# cv2.imshow("g",gray)
rows = gray.shape[0]

# Applying hough 
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                           param1=100, param2=30,
                           minRadius=10, maxRadius=250)

if circles is not None:
    circles = np.uint16(np.around(circles))
    ctr = 0
    for i in circles[0, :]:
        ctr+=1
        # Finding nearest circle to the median 
        print(f"circle {ctr} points: " + str(i), end="")
        dist = abs(math.sqrt((k-i[0])**2 + (l-i[1])**2))
        print(" dist: " + str(dist))
        if dist<curr_dist or curr_dist == 0:
            nearest_circle = i
            curr_dist = dist
            dist = 0
        # numbering circle
        cv2.putText(gray, str(ctr), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (246,255,12), 3)    
        # draw circle 
        radius = i[2]
        #cv2.circle(gray, (i[0],i[1]), radius, (255, 0, 255), 3)

print(f"nearest circle: {nearest_circle}")            

# c1 and c2 are the center of the hough circle
c1,c2=nearest_circle[0], nearest_circle[1]    
side=(nearest_circle[2])*2.5
side=int(side)

# draw median on final image
# cv2.circle(img_final, (k,l), 1, (255, 0, 255), 10)
# draw circle on final image
# cv2.circle(img_final, (c1,c2), nearest_circle[2], (255, 0, 255), 3)

p1=(c1-side,c2-side)
p2=(c1+side,c2-side)
p3=(c1+side,c2+side)
p4=(c1-side,c2+side)

a,b=gray.shape
l=[]
for i in range(a):
    for j in range(b):
        if gray[i][j]==255 and i>(c1-side) and i<(c1+side) and j>(c2-side) and j<(c2+side):
            l.append([i,j])

# cv2.imshow("matches", img3)
# cv2.imshow("detected circles", gray)
# cv2.waitKey(0)

# PCA
X, y = make_classification(n_samples=1000)
X=np.array(l)
n_samples = X.shape[0]

# create a PCA object
pca = PCA()
X_transformed = pca.fit_transform(X)

# We center the data and compute the sample covariance matrix.
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca.explained_variance_
eigenvectors=[]
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    # print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    # print(f"eigenValue: {eigenvalue}")
    eigenvectors.append(eigenvector)

print(f"eigenVectors: {eigenvectors}")
tampp = 45
angle2=math.atan2(eigenvectors[1][1],eigenvectors[1][0])*180/math.pi+tampp
angle1=math.atan2(eigenvectors[0][1],eigenvectors[0][0])*180/math.pi+tampp

# for drawing rotation angle as found from PCA
centre_tuple = (c1,c2)
endpoint1 = (c1 + np.uint16(eigenvectors[0][0]*100), c2 + np.uint16(eigenvectors[0][1]*100))
# cv2.line(img_final,centre_tuple, endpoint1, (0, 0, 255))
# cv2.line(img2,centre_tuple, (c1+side,c2), (0, 255, 0)) 

angle_deg=min(angle1,angle2)
print(f"angle_deg: {angle_deg}")

if angle_deg<180:
    theta = angle_deg+180 
else:
    theta = angle_deg    

# scaling for bounding box
scale_top = 1.5*2
scale_bottom = 4.5*3
scale_side = 8

# Find bounding box corner points which is scaled and rotated 
A = [c1+(scale_top*radius), c2+(scale_side*radius)]
B = [c1-(scale_bottom*radius), c2+(scale_side*radius)]
C = [c1-(scale_bottom*radius), c2-(scale_side*radius)]
D = [c1+(scale_top*radius), c2-(scale_side*radius)] 

hyp_top = round(abs(math.sqrt((A[0]-c1)**2 + (A[1]-c2)**2)))
hyp_bottom = round(abs(math.sqrt((B[0]-c1)**2 + (B[1]-c2)**2)))

A_offset = math.degrees(math.atan(scale_side/scale_top))
B_offset = math.degrees(math.atan(scale_bottom/scale_side))
C_offset = 90-B_offset
D_offset = 90-A_offset

# four corner points of bounding box after scaling and rotating
A_new = [round(c1+hyp_top*math.cos(math.radians(A_offset+theta))), 
         round(c2+hyp_top*math.sin(math.radians(A_offset+theta)))]
B_new = [round(c1-hyp_bottom*math.sin(math.radians(B_offset+theta))), 
         round(c2+hyp_bottom*math.cos(math.radians(B_offset+theta)))]
C_new = [round(c1-hyp_bottom*math.cos(math.radians(C_offset+theta))), 
         round(c2-hyp_bottom*math.sin(math.radians(C_offset+theta)))]
D_new = [round(c1+hyp_top*math.sin(math.radians(D_offset+theta))), 
         round(c2-hyp_top*math.cos(math.radians(D_offset+theta)))]

# labeling four corners of bounding box
cv2.putText(img_final, "A", (A_new[0]-20,A_new[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.putText(img_final, "B", (B_new[0]-20,B_new[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.putText(img_final, "C", (C_new[0]-20,C_new[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.putText(img_final, "D", (D_new[0]-20,D_new[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# Draw bounding box
contour = np.array([A_new, B_new, C_new, D_new])
cv2.drawContours(img_final,[contour],0,(0,0,255),10)
cv2.imwrite("Rot.jpeg",img_final)

# creating an image copy for mask
img_ = np.copy(img_final)

# all locations outside bounding box made black 
for i in range(img_final.shape[0]):
    for j in range(img_final.shape[1]):
        if cv2.pointPolygonTest(contour, (j, i), False) == -1.0:
            #print(f"{i} {j}")
            img_[i][j] = np.array([0, 0, 0])
            

# cv2.imshow("detected circles: ", gray)
cv2.imshow("result", img_final)
cv2.imwrite("result_"+img_name, img_final)
cv2.imshow("contour_space", img_)
cv2.imwrite("contour_space_"+img_name, img_)

# Applying GaussianBlur and Converting Image into HSV colorspace
gaus=cv2.GaussianBlur(img_,(25,25),cv2.BORDER_DEFAULT)
hsv_conv=cv2.cvtColor(gaus,cv2.COLOR_BGR2HSV)

# threshold values for HSV
lower_bound_body = np.array([31,1,1])
upper_bound_body = np.array([180,255,255])
lower_bound_face = np.array([0,60,68])
upper_bound_face = np.array([18,88,255])
lower_bound_beard = np.array([0,115,115])
upper_bound_beard = np.array([35,255,255])

# create masks
mask1 = cv2.inRange(hsv_conv,lower_bound_body,upper_bound_body)
mask2 = cv2.inRange(hsv_conv,lower_bound_face,upper_bound_face)
mask3 = cv2.inRange(hsv_conv,lower_bound_beard,upper_bound_beard)

# bitwise OR to get the overall mask
final_mask = mask1|mask2|mask3

#Inverse mask for the background
inv_mask=cv2.bitwise_not(final_mask)

#Segment Object Using Mask
masked_img_object=cv2.bitwise_and(img_final,img_final,mask=final_mask)

#Segment Background Using Inverse Mask
masked_img_background=cv2.bitwise_and(img_final,img_final,mask=inv_mask)

#Show all the masked images
cv2.imshow("segment1",masked_img_object)
cv2.imwrite("segment1_"+img_name,masked_img_object)
cv2.imshow("segment",masked_img_background)
cv2.imwrite("segment_"+img_name,masked_img_background)

#Add the only object and only background image in a weighted ration of 8:2
img_segment=cv2.addWeighted(masked_img_object, 0.8, masked_img_background,
0.2,0)

# Draw the final bounding box using red color
cv2.drawContours(img_segment,[contour],0,(0,0,255),4)

# Show the final image
cv2.imshow("Final Answer",img_segment)

#Write the final image to file
cv2.imwrite("Answer_"+img_name,img_segment)
cv2.imshow("final mask",final_mask)
cv2.imwrite("final_mask_"+img_name,final_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""##################################################################################################
#Mask R-CNN
import cv2
import numpy as np
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

#matplotlib inline
def random_colors(N): #Random Colors of Masks
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):# Apply the mask to the results

    #apply mask to image
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1,image[:, :, n] * (1 - alpha) + alpha * c,image[:, :, n])
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    
    #take the image and results and apply the mask, box, and Label
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    
    for i, color in enumerate(colors):    
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX,0.7, color, 2)
    return image

if __name__ == '__main__':
    #test everything
    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")
    
    # Import Mask RCNN
    sys.path.append(ROOT_DIR) # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    
    # Import config
    sys.path.append(os.path.join(ROOT_DIR, "samples/")) # To find local version
    
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "plushies.h5")# load the weights
    
    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images") #image directory if testing on images
    
    class InferenceConfig(plushie.Config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1 # As we have only 8GB VRAM on GTX 1070 it is set to 1
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 0.6 Use only 60% VRAM
        keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    
    config = InferenceConfig()
    config.display()
        
    #Run MaskRCNN model in Inference Mode
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(MODEL_PATH, by_name=True)
    class_names = ['plushies']
    capture = cv2.VideoCapture(r"E:\cvproject\2021-12-13 05-19-55.mp4")
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter(r'E:\cvproject\Mask_RCNN\samples\videofile_masked.avi', codec, 60.0,size)
    
    while (capture.isOpened()):
        ret, frame = capture.read()#Read the video frame by frame
        if ret:
            results = model.detect([frame], verbose=0)
            r = results[0]
            frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])# Perform Instance Segmentation on each image
            output.write(frame) # Write out the frame into a video file
        else:
            break
    
    capture.release()
    cv2.destroyAllWindows()"""
