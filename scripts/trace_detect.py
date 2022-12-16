import cv2 as cv
import numpy as np
import time, random

# read board scan image
board_img = cv.imread("/home/maximus/Xbox/board_scan/1.6/xbox_1_6_top_crop.png")

print('Original Dimensions : ',board_img.shape)
 
scale_percent = 15 # percent of original size
width = int(board_img.shape[1] * scale_percent / 100)
height = int(board_img.shape[0] * scale_percent / 100)
dim = (width, height)
   

# convert to HSV
board_hsv = cv.cvtColor(board_img, cv.COLOR_BGR2HSV)

def perc2int(val):
    return int(val * 255 / 100)
def Hperc2int(val):
    return int(val // 2)

def get_trace_mask():
    # Traces
    # sample: H: 92.7 S: 44.0 V: 39.2
    # sample: H: 95.1 S: 37.3 V: 43.1
    # sample: H: 62.5 S: 38.7 V: 48.6
    # sample: H: 75.3 S: 52.6 V: 38.0

    # sample: H: 55.8 S: 29.8 V: 74.9

    # Fill
    # sample: H: 148.6 S: 79.2 V: 41.6

    # Negative:
    # sample: H: 171.8 S: 89.5 V: 22.4
    # sample: H: 171.4 S: 92.5 V: 20.8

    lower_threshold = np.array([Hperc2int(62),
                                perc2int(20),
                                perc2int(20)])
    
    higher_threshold = np.array([Hperc2int(120),
                                 perc2int(100),
                                 perc2int(100)])

    trace_mask = cv.inRange(board_hsv, lower_threshold, higher_threshold)

    # more copper tone
    lower_threshold = np.array([Hperc2int(38),
                                perc2int(18),
                                perc2int(18)])
    
    higher_threshold = np.array([Hperc2int(65),
                                 perc2int(100),
                                 perc2int(100)])

    trace_mask2 = cv.inRange(board_hsv, lower_threshold, higher_threshold)

    return trace_mask + trace_mask2


#del(board_img)
trace_mask = get_trace_mask()

#output_img = cv.bitwise_and(board_img, board_img, mask=trace_mask)


# resize image
#trace_mask = cv.resize(trace_mask, dim, interpolation = cv.INTER_AREA)

# cv.imshow("Output", trace_mask)
# cv.waitKey(0)

# fill in holes
kernel = cv.getStructuringElement(cv.MORPH_RECT,(4,4))
trace_filter = cv.morphologyEx(trace_mask, cv.MORPH_CLOSE, kernel)
#kernel = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
#trace_filter = cv.morphologyEx(trace_filter, cv.MORPH_OPEN, kernel)

mask = np.zeros(trace_filter.shape[:2], dtype=np.uint8)

contours, hierarchy = cv.findContours(trace_filter, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE) # Use cv2.CCOMP for two level hierarchy
# loop through the contours
for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it
    #if hierarchy[0][i][3] != -1: # basically look for holes
    # if the size of the contour is less than a threshold (noise)
    if cv.contourArea(cnt) < 100:
        # Fill the holes in the original image
        cv.drawContours(trace_filter, [cnt], 0, (0), -1)

trace_thin = cv.ximgproc.thinning(trace_mask,0)

# contours, hierarchy=cv.findContours(trace_thin.clone(),cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
# for c in contours:
#     l = cv.arcLength(c, True)
#     a = cv.contourArea(c)
#     if l < 20:
#         continue
#     if a > l:
#         continue
#     #eps = 0.005
#     #approx = cv.approxPolyDP(c, eps * l, True)
#     cv.drawContours(mask, [c], 0, (255), 1)


retval, labels = cv.connectedComponents(trace_thin)

# ##################################################
ts = time.time()
num = labels.max()

rand_hue = np.random.randint(179, size=num+1)
rand_hue[0] = 0
label_hue = np.uint8(rand_hue[labels])

# label_hue = np.zeros(trace_filter.shape[:2], dtype=np.uint8)

# N = 50
# for i in range(1, num+1):
#     pts =  np.where(labels == i)
#     if len(pts[0]) < N:
#         continue
#     rand_hue = random.randrange(180)
#     label_hue[pts] = rand_hue

# #         labels[pts] = 0

print("Time passed: {:.3f} ms".format(1000*(time.time()-ts)))
# # Time passed: 4.607 ms

# ##################################################

# Map component labels to hue val
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

cv.imwrite("out_mask.jpg", labeled_img)
# cv.imwrite("out_thin.jpg", trace_thin)
# cv.imwrite("out.jpg", trace_filter)