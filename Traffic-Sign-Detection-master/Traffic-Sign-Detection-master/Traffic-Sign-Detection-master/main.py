import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math

from classification import training, getLabel

SIGNS = ["ERROR",
         "STOP",
         "TURN LEFT AHEAD",
         "TURN RIGHT AHEAD",
         "DO NOT TURN LEFT",
         "DO NOT TURN RIGHT",
         "ONE WAY",
         "SPEED LIMIT",
         "MINIMUM SPEED LIMIT",
         "TURN RIGHT",
         "NATIONAL SPEED LIMIT",
         "END OF MINIMUM SPEED",
         "TURN LEFT"]

# Clean all previous file
#hàm xóa toàn bộ các ảnh chụp khi chạy lại
def clean_images():
	file_list = os.listdir('./')
	for file_name in file_list:
		if '.png' in file_name:
			os.remove(file_name)


### Preprocess image: xử lí hình ảnh
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  #chuyển từ BGR sang YCrCb
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)      #làm mờ ảnh     # paramter
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)  #chuyển qua ảnh đa cấp xám
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)  #làm rõ biên     # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image) #giãn ảnh, lấy gtri tuyệt đối, chuyển sang 8 bit k dấu
    return LoG_image
    
def binarization(image): #ảnh trắng đen(ảnh nhị phân)
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1] #để lấy hình ảnh hai cấp (nhị phân) ra khỏi hình ảnh thang độ xám
    return thresh

def preprocess_image(image): #tiến trình xử lý ảnh
    image = constrastLimit(image) #giới hạn độ tương phản
    image = LaplacianOfGaussian(image) #lọc biên
    image = binarization(image) #chuyển sang ảnh trắng đen để dễ xử lý
    return image

# Find Signs
def removeSmallComponents(image, threshold): #loại bỏ nhiễu dựa vào ngưỡng ảnh
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    #find contours in the thresholded image: tìm đường viền trong ngưỡng hình ảnh
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE    )
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return cnts

#ktra xem đường viền có phải của biển báo k
def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold: chu vi, tâm, ngưỡng
    # # Compute signature of contour
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

#crop sign :
def cropContour(image, center, max_distance):
    # cắt đường viền
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    print(left, right, top, bottom)
    return image[left:right, top:bottom]

def cropSign(image, coordinate):
    #  cắt biển báo dựa vào hình và tọa độ
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]

# vẽ khung bao biển báo
def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            sign = cropSign(image,coordinate)
    return sign, coordinate




def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)

    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))

    #binary_image = remove_line(binary_image)

    cv2.imshow('BINARY IMAGE', binary_image)
    contours = findContour(binary_image)

    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    
    text = ""
    sign_type = -1
    i = 0
    # nhận dạng loại biển báo
    if sign is not None:
        #cắt hình là biển báo sau đó lưu vào file dạng số_tên.png
        sign_type = getLabel(model, sign)
        sign_type = sign_type if sign_type <= 11 else 11
        text = SIGNS[sign_type]
        cv2.imwrite(str(count)+'_'+text+'.png', sign)

    if sign_type > 0 and sign_type != current_sign_type:    #vẽ hình chữ nhật bao bên ngoài và puttext(ghi tên biển báo)
        cv2.rectangle(original_image, coordinate[0],coordinate[1], (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(original_image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
    return coordinate, original_image, sign_type, text



def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    return mask #mặt nạ để che biển báo, cắt các đối tượng bên ngoài

def main(args):
	#Clean previous image    
    clean_images()
    #Training phase
    model = training()

    vidcap = cv2.VideoCapture(args.file_name)

#lấy khung hình mỗi giây và nhận dạng khung hình đó
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  # float
    height = vidcap.get(4) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #nén file  có kích thước lớn về dạng XVID
    out = cv2.VideoWriter('output.avi',fourcc, fps , (640,480))

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    success = True
    similitary_contour_with_circle = 0.65   # parameter
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []
    file = open("Output.txt", "w") #mở file output
    while True:
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        #frame = cv2.resize(frame, (640,int(height/(width/640))))
        frame = cv2.resize(frame, (640,480))

        print("Frame:{}".format(count))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        coordinate, image, sign_type, text = localization(frame, args.min_size_components, args.similitary_contour_with_circle, model, count, current_sign)

        if coordinate is not None:
            cv2.rectangle(image, coordinate[0],coordinate[1], (255, 255, 255), 1)
        print("Sign:{}".format(sign_type))
        if sign_type > 0 and (not current_sign or sign_type != current_sign):
            current_sign = sign_type
            current_text = text
            top = int(coordinate[0][1]*1.05)
            left = int(coordinate[0][0]*1.05)
            bottom = int(coordinate[1][1]*0.95)
            right = int(coordinate[1][0]*0.95)

            position = [count, sign_type if sign_type <= 8 else 8, coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]]
            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)

            tl = [left, top]
            br = [right,bottom]
            print(tl, br)
            current_size = math.sqrt(math.pow((tl[0]-br[0]),2) + math.pow((tl[1]-br[1]),2))
            # grab the ROI for the bounding box and convert it
            # to the HSV color space
            try:
                roi = frame[tl[1]:br[1], tl[0]:br[0]]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

                # compute a HSV histogram for the ROI and store the
                # bounding box
                roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
                roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                roiBox = (tl[0], tl[1], br[0], br[1])
            except Exception:
                pass

        elif current_sign:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

            # apply cam shift to the back projection, convert the
            # points to a bounding box, and then draw them
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.boxPoints(r))
            s = pts.sum(axis = 1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            size = math.sqrt(pow((tl[0]-br[0]),2) +pow((tl[1]-br[1]),2))
            print(size)

            if  current_size < 1 or size < 1 or size / current_size > 30 or math.fabs((tl[0]-br[0])/(tl[1]-br[1])) > 2 or math.fabs((tl[0]-br[0])/(tl[1]-br[1])) < 0.5:
                current_sign = None
                print("Stop tracking")
            else:
                current_size = size

            if sign_type > 0:
                top = int(coordinate[0][1])
                left = int(coordinate[0][0])
                bottom = int(coordinate[1][1])
                right = int(coordinate[1][0])

                position = [count, sign_type if sign_type <= 8 else 8, left, top, right, bottom]
                cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
            elif current_sign:
                position = [count, sign_type if sign_type <= 8 else 8, tl[0], tl[1], br[0], br[1]]
                cv2.rectangle(image, (tl[0], tl[1]),(br[0], br[1]), (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image,current_text,(tl[0], tl[1] -15), font, 1,(0,0,255),2,cv2.LINE_4)

        if current_sign:
            sign_count += 1
            coordinates.append(position)

        cv2.imshow('Result', image)
        count = count + 1
        #Write to video
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    file.write("{}".format(sign_count))
    for pos in coordinates:
        file.write("\n{} {} {} {} {} {}".format(pos[0],pos[1],pos[2],pos[3],pos[4], pos[5]))
    print("Finish {} frames".format(count))
    file.close()
    return 


if __name__ == '__main__':#để chạy trong python console
    parser = argparse.ArgumentParser(description="NLP Assignment Command Line")
    
    parser.add_argument(
      '--file_name',
      default= "./QuestionVideo.avi",
      help= "Video to be analyzed"
      )
    
    parser.add_argument(
      '--min_size_components',
      type = int,
      default= 300,
      help= "Min size component to be reserved"
      )

    
    parser.add_argument(
      '--similitary_contour_with_circle',
      type = float,
      default= 0.65,
      help= "Similitary to a circle"
      )
    
    args = parser.parse_args()
    main(args)
