# import wget

# wget.download('https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml')

import cv2
import sys
import os.path
import os

def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    imgNum  = 0

    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cropped = image[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
        # cropped = image[y - int(h/16):y + h + int(h/16), x - int(w/16):x + w + int(w/16)]
        cropped = image[y:y+h, x:x+w]

        if (x,y,w,h) in faces is ():
            print("Face not Found")
            pass            
        
        else :
            cv2.imwrite("10_%s"%i, cropped)
            # cv2.imwrite("zz.jpg", cropped)


    # cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("out", cropped)


# detect('C:/Users/JISUP LIM/Desktop/aninme/1/270.jpg')
 

workDIr = os.path.abspath('C:/Users/Admin/Desktop/anime/new5')
for dirpath, dirnames, filenames in os.walk(workDIr):

    for filename in filenames:
        # print("\t", filename)
        for i in filenames :
            detect("C:/Users/Admin/Desktop/anime/new5/%s"%i,)



# import cv2
# import sys
# import os.path

# def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
#     if not os.path.isfile(cascade_file):
#         raise RuntimeError("%s: not found" % cascade_file)

#     cascade = cv2.CascadeClassifier(cascade_file)
#     image = cv2.imread(filename, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
    
#     faces = cascade.detectMultiScale(gray,
#                                      # detector options
#                                      scaleFactor = 1.1,
#                                      minNeighbors = 5,
#                                      minSize = (24, 24))
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

#     cv2.imshow("AnimeFaceDetect", image)
#     cv2.waitKey(0)
#     cv2.imwrite("zz.png", image)

# if len(sys.argv) != 2:
#     sys.stderr.write("usage: detect.py <filename>\n")
#     sys.exit(-1)
    
# detect(sys.argv[1])