import numpy as np
import cv2

folder_path = 'D:/Project/'

cap1 = cv2.VideoCapture(folder_path + '1.mp4')
cap2 = cv2.VideoCapture(folder_path + '2.mp4')
cap3 = cv2.VideoCapture(folder_path + '3.mp4')
cap4 = cv2.VideoCapture(folder_path + '4.mp4')
car=cv2.imread(folder_path + 'car.png')

while(cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened()):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()


    mask1 = np.zeros(frame1.shape, dtype=np.uint8)
    roi_corners1 = np.array([[(1,1), (1920,1), (960,1080)]], dtype=np.int32)
    white1 = (255, 255, 255)
    cv2.fillPoly(mask1, roi_corners1, white1)
    masked_image1 = cv2.bitwise_and(frame1, mask1)
    masked_image1 = cv2.resize(masked_image1,(300,460))
    (rows1, cols1) = masked_image1.shape[:2]


    mask2 = np.zeros(frame2.shape, dtype=np.uint8)
    roi_corners2 = np.array([[(1,1), (1920,1), (960,1080)]], dtype=np.int32)
    white2 = (255, 255, 255)
    cv2.fillPoly(mask2, roi_corners2, white2)
    masked_image2 = cv2.bitwise_and(frame2, mask2)
    masked_image2 = cv2.resize(masked_image2,(420,420))
    (rows2, cols2) = masked_image2.shape[:2]
    M2 = cv2.getRotationMatrix2D((cols2/2,rows2/2),270,1)
    masked_image2 = cv2.warpAffine(masked_image2,M2,(cols2,rows2))


    mask3 = np.zeros(frame3.shape, dtype=np.uint8)
    roi_corners3 = np.array([[(1,1), (1920,1), (960,1080)]], dtype=np.int32)
    white3 = (255, 255, 255)
    cv2.fillPoly(mask3, roi_corners3, white3)
    masked_image3 = cv2.bitwise_and(frame3, mask3)
    masked_image3 = cv2.resize(masked_image3,(300,460))
    (rows3, cols3) = masked_image3.shape[:2]
    M3 = cv2.getRotationMatrix2D((cols3/2,rows3/2),180,1)
    masked_image3 = cv2.warpAffine(masked_image3,M3,(cols3,rows3))


    mask4 = np.zeros(frame4.shape, dtype=np.uint8)
    roi_corners4 = np.array([[(1,1), (1920,1), (960,1080)]], dtype=np.int32)
    white4 = (255, 255, 255)
    cv2.fillPoly(mask4, roi_corners4, white4)
    masked_image4 = cv2.bitwise_and(frame4, mask4)
    masked_image4 = cv2.resize(masked_image4,(420,420))
    (rows4, cols4) = masked_image4.shape[:2]
    M4 = cv2.getRotationMatrix2D((cols4/2,rows4/2),90,1)
    masked_image4 = cv2.warpAffine(masked_image4,M4,(cols4,rows4))
    

    numpy_vertical = np.vstack((masked_image1, masked_image3))
    numpy_horizontal = np.hstack((masked_image4, masked_image2))
    numpy_vertical = cv2.resize(numpy_vertical,(400,720))
    numpy_horizontal = cv2.resize(numpy_horizontal,(400,720))

    numpy_verticali1 = np.vstack((frame1,frame3))
    numpy_verticali2 = np.vstack((frame2,frame4))
    numpy_horizontali1 = np.hstack((numpy_verticali1,numpy_verticali2))


    foreground, background = numpy_vertical.copy(), numpy_horizontal.copy()
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    alpha = 0.5
    blended_portion = cv2.addWeighted(foreground,alpha,background[:foreground_height,:foreground_width,:],1-alpha,0,background)
    background[:foreground_height,:foreground_width,:] = blended_portion

    car = cv2.resize(car,(600,600))
    background = cv2.resize(background,(600,600))
    background = cv2.addWeighted(background, 2.0 , car, 1.0, 1.0)
    numpy_horizontali1 = cv2.resize(numpy_horizontali1,(600,600))
    final_image = np.hstack((background,numpy_horizontali1))

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfTextv = (120,20)
    bottomLeftCornerOfText1 = (600,20)
    bottomLeftCornerOfText2 = (900,20)
    bottomLeftCornerOfText3 = (600,320)
    bottomLeftCornerOfText4 = (900,320)
    fontScale              = 0.7
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(final_image,'Virtual Aerial View Projection',bottomLeftCornerOfTextv,font,fontScale,fontColor,lineType)
    cv2.putText(final_image,'Camera 1',bottomLeftCornerOfText1,font,fontScale,fontColor,lineType)
    cv2.putText(final_image,'Camera 2',bottomLeftCornerOfText2,font,fontScale,fontColor,lineType)
    cv2.putText(final_image,'Camera 3',bottomLeftCornerOfText3,font,fontScale,fontColor,lineType)
    cv2.putText(final_image,'Camera 4',bottomLeftCornerOfText4,font,fontScale,fontColor,lineType)


    cv2.imshow('masked image', final_image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
