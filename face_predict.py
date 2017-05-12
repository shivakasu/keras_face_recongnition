#-*- coding: utf-8 -*-

import cv2
from face_train import Model

if __name__ == '__main__':
        
    #加载模型
    model = Model()
    model.load_model(file_path = 'face_model_re.h5')    
    
    color = (0, 255, 0)    
    cap = cv2.VideoCapture(0)
    cascade_path = "E:\cs\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml"    
    
    #循环检测识别人脸
    while True:
        _, frame = cap.read()   #读取一帧视频
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)   
                textf = ""
                if faceID == 0:
                    textf = "wkx"
                elif faceID == 1:
                    textf = "zcw" 
                elif faceID == 2:
                    textf = "whm" 
                else:
                    textf = "unknow"
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                cv2.putText(frame,textf, 
                            (x + 30, y + 30),                      #坐标
                            cv2.FONT_HERSHEY_SIMPLEX,              #字体
                            1,                                     #字号
                            (255,0,255),                           #颜色
                            2)                                     #字的线宽
                    
                            
        cv2.imshow("face predict", frame)
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()