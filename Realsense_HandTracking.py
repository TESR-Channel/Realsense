import cv2
import pyrealsense2 as rs
import numpy as np

#confing resolution of realsense
rs_w=640
rs_h=360
fps=60
#  1280x720 fps 30, 848x480 fps60, 640x480 fps60, 640x360 fps60 , 424x240 fps60

#Start realsense pipeline 
pipeline= rs.pipeline()
config= rs.config()
#Eneble device id for more than 1 realsense
#config_1.enable_device('018322070394')

#eneble video stream color and depth
config.enable_stream(rs.stream.depth, rs_w, rs_h, rs.format.z16, fps)
config.enable_stream(rs.stream.color, rs_w, rs_h, rs.format.bgr8, fps)
pipeline.start(config)

#funtion for easily concate video or image
def concat_tile(im_list_2d):
	return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

#prevent divided by 0
def safe_div(x, y):  
	if y == 0: return 0
	return x / y

#Function to find mid point
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#parameter of realsense
def ALPHA(al):
    if al > 3:
        a = (al * (2 / 100))
    else:
        a = 0.009
    return a
# for trackbar
def nothing(x):  
    pass

#name window that show output
windowName="TEST"
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.createTrackbar("threshold", windowName, 45, 255, nothing)
cv2.createTrackbar("kernel", windowName, 4, 30, nothing)
cv2.createTrackbar("iterations", windowName, 2, 10, nothing)
cv2.createTrackbar("alpha", windowName, 20, 100, nothing)

showLive=True

#set font of all character
font=cv2.FONT_HERSHEY_SIMPLEX

while (showLive):
	
    ### Setting thresh kern itera
	thresh = cv2.getTrackbarPos("threshold", windowName)
	kern = cv2.getTrackbarPos("kernel", windowName)
	itera = cv2.getTrackbarPos("iterations", windowName)
	Alp = cv2.getTrackbarPos("alpha", windowName)

	#wait for realsense frame input. if not it will crash out
	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()

	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())

	#rearrange depth_image to color image
	depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha= ALPHA(Alp)), cv2.COLORMAP_JET)

	#Convert depth to gray scale
	gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
	
	#threshold operating
	ret, thresh = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

	#kernel setting
	kernel = np.ones((kern, kern), np.uint8)


	dilation = cv2.dilate(thresh, kernel, iterations=itera)
	erosion = cv2.erode(thresh, kernel, iterations=itera)

	###select edge refined
	edge = dilation

	opening = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(edge,cv2.MORPH_CLOSE, kernel)

    ###select Filter
	Morphological = closing

    #Contour in filtered image
	contours, hierarchy = cv2.findContours(Morphological, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #Check if have contour then contunue. If not break
	if len(contours)!=0:
		
		areas = []  # list to hold all areas

		#find hightest contour
		for contour in contours:
			ar = cv2.contourArea(contour)

			areas.append(ar)
		max_area_index = areas.index(max(areas))
		cnt = contours[max_area_index]

		#Convex hull for hand detection
		hull=cv2.convexHull(cnt)
		hull_defect = cv2.convexHull(cnt, returnPoints=False)
		defects = cv2.convexityDefects(cnt, hull_defect)
		if defects is not None:
			count = 0

			# calculate the angle
			for i in range(defects.shape[0]):  
				s, e, f, d = defects[i][0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])
				a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
				b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
				c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
				angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

				# angle less than 90 degree, treat as fingers
				if angle <= np.pi / 2: 
					count += 1
					cv2.circle(depth_colormap, far, 4, [0, 0, 255], -1)
			if count > 0:
				count = count+1
				
			#put finger counting in to image
			cv2.putText(depth_colormap, str(count), (rs_w-50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0) , 2, cv2.LINE_AA)
		cv2.drawContours(depth_colormap, [hull], -1, (0,255,0), 3)

		cv2.drawContours(depth_colormap, [cnt], -1, (0,0,255), 2)
	Morphological = cv2.cvtColor(Morphological, cv2.COLOR_GRAY2BGR)

	#concate color image and depth image
	show=concat_tile(([color_image,depth_colormap],[Morphological,Morphological]))
	
	#show output image
	cv2.imshow('TEST', show)

	#wait key for exit
	key = cv2.waitKey(10)
	if key & 0xFF == ord('q') or key == 27:
		showLive = False
		break

pipeline.stop()
cv2.destroyAllWindows()