import cv2

def width(url):
    vcap = cv2.VideoCapture(url)  # 0=camera
    width = int(vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))  #`width`
    return width

def height(url):
        vcap = cv2.VideoCapture(url)  # 0=camera
        height = int(vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))  #`height`
        return height