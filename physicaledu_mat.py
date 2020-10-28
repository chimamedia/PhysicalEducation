import cv2
import argparse
import chainer
from entity import params
from pose_detector import PoseDetector, draw_person_pose
from face_detector import FaceDetector, draw_face_keypoints
from hand_detector import HandDetector, draw_hand_keypoints

import numpy as np
from time import sleep
from tqdm import tqdm
import sys, time

import os

chainer.using_config('enable_backprop', False)

Colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        
if __name__ == '__main__':
    tp = lambda x:list(map(int, x.split('.')))
    
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--video', help='video file path')
    parser.add_argument('--gpu', '-g', type=int, default = -1, help = 'GPU ID (negative value indicates CPU)')
    parser.add_argument('--point', '-p', type=int, default = -1, help = 'Plot skeleton node point (0 - 17).')
    parser.add_argument('--draw_point', '-dp', default = [], type=tp, help = 'Plot skeleton node points array (0 - 17). example: 0.1.2')
    parser.add_argument('--mode', '-m', type=int, default = 0, help = 'zenten = 0, kouten = 1')
    parser.add_argument('--out', default='result', help='output video name.')
    args = parser.parse_args()
    
    point = args.point
    if point > 17:
        point = 17
    if point < -1:
        point = -1
    
    for i in args.draw_point:
        if i > 17:
            i = 17
            
    if point > 17:
        print("draw point: " + str(point))
    elif len(args.draw_point) > 0:
        print("draw point: ", end = "")
        print(args.draw_point)
    else:
        print("draw point: ALL")

    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    # hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)
    # face_detector = FaceDetector("facenet", "models/facenet.npz", device=args.gpu)
    
    # read video
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.out + '_skeleton.avi',fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(args.out + '_skeleton-view.avi',fourcc, fps, (width, height))
    count = 0
    
    all_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT);
    pbar = tqdm(total=all_frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font2 = cv2.FONT_HERSHEY_COMPLEX
    
    flag1 = -1
    flag2 = -1
    flag3 = -1
    flag4 = 1
    flag5 = -1
    
    while True:
        
        #get video frame
        ret, img = cap.read()
        
        if not ret:
            print("Failed to capture image")
            break
        
        # 跳び箱の基準線
        h, w, c = img.shape
        hw = 0.68
        hline = h * hw
        cv2.line(img, (0, int(hline)), (w, int(hline)), (0, 0, 255), thickness=3, lineType=cv2.LINE_4)
        
        person_pose_array, _ = pose_detector(img)
        if(count == 0):
            image = img.copy()
        
        #前転の場合
        if(args.mode == 0):
            
            
            for pose in person_pose_array.round().astype('i'):
                if (len(args.draw_point) == 0):
                    
                    #flag_first: 手をついたときを判定
                    flag_first = True
                    rhandX = 0
                    lhandX = 10e9
                    
                    points = []
                    for i, (x, y, v) in enumerate(pose):
                        
                        if v != 0:
                            points.append((x, y))
                            cv2.circle(image, (x, y), 2, Colors[i], -1)
                        else:
                            points.append((-1, -1))
                        
                    
                    #両手をつく(両手が基準線より下で、それぞれの距離が25ピクセル以下（平行）)
                    if(points[4][1] > int(hline) and points[7][1] > int(hline) \
                       and (abs(points[4][0] - points[7][0]) < 25 \
                       and points[4][0] > 0 and points[7][0] > 0  )):
                    	cv2.putText(img,'Good (Put Hand)',(700,500),font,2,(255,200,0),cv2.LINE_AA)
                    	cv2.putText(image,'Good (Put Hand)',(700,500),font,2,(255,200,0),cv2.LINE_AA)
                    
                    #両足がジャンプしているときに腰が上にあるか
                    #cv2.circle(img, points[10], 5, Colors[10], -1)
                    #cv2.circle(img, points[13], 5, Colors[13], -1)
                    #cv2.circle(img, points[8], 5, Colors[8], -1)
                    #cv2.circle(img, points[11], 5, Colors[11], -1)
                    
                    if(points[10][1] < int(hline) and points[13][1] < int(hline) \
                       and points[10][0] > 0 and points[13][0] > 0 ):
                        
                        #cv2.putText(img, (str(points[10][1]) + '  ' + str(points[13][1])),(800,300),font,2,(255,125,0),cv2.LINE_AA)
                        
                        local_max = points[8][1] #右付け根を最大
                        flag_hip = True
                        for i in range(17):
                    	    if i != 11: #左付けねは除外
                    	        if local_max > points[i][1] and points[i][1] > 0:
                    	            flag_hip = False
                    	            break
                        if(flag_hip == True):
                            cv2.putText(img,'Good (High Hip)',(800,400),font,2,(255,125,0),cv2.LINE_AA)
                            cv2.putText(image,'Good (High Hip)',(800,400),font,2,(255,125,0),cv2.LINE_AA)
                            
                        
                    #最後に静止しているかどうか
                    
                
                else:
                    for j in args.draw_point:
                        #両手が基準線より下（マットに手をついた）
                        for i, (x, y, v) in enumerate(pose):
                            if(j == i or j < 0):
                                cv2.circle(image, (x, y), 5, Colors[i], -1)
                
            
            


        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)        
        out.write(res_img)
                    
#                print(i)
#                break
                #if v != 0:
                    #cv2.circle(canvas, (x, y), 3, color, -1)
                #    print(x,"-",y)
                #    cv2.circle(image, (x, y), 10, (255, 0, 0), -1)
            

        #cv2.imshow('image',img)
        #cv2.waitKey(1)
        out2.write(cv2.addWeighted(img, 0.6, image, 0.4, 0))
        count+=1
        
        pbar.update(1)
        
    #ファイル出力
    basename = os.path.basename(args.video)
    path_w = basename + ".txt"
    with open(path_w, mode='w') as f:
        f.write(str(flag1) + "," + str(flag2) + "," + str(flag3) + "," + str(flag4) + "," + str(flag5))
        
        
        # 以下は不要
        '''
        #2020年追加評価箇所
        #1 踏切
        img = cv2.rectangle(img,(1485,695),(1782,845),(255,0,0),3)
        
        #2 台上
        img = cv2.rectangle(img,(704,360),(921,486),(0,255,0),3)
        
        #3 着地
        img = cv2.rectangle(img,(124,785),(622,949),(0,0,255),3)
        
        for pose in person_pose_array.round().astype('i'):
            head = 0
            hip = 0
            handl = 10e9
            legl = 10e9
            handr = 10e9
            legr = 10e9
            #print("pose")
            #print(pose)
            
            height, width = img.shape[:2]
            if (len(args.draw_point) == 0):
                for i, (x, y, v) in enumerate(pose):
                    if(point == i or point < 0):
                        cv2.circle(image, (x, y), 5, Colors[i], -1)
                    
                    if (i==10 or i==13) and (1485 < x < 1782 and 695 < y < 845):
                        cv2.putText(image,'Good',(1200,500),font,4,(255,0,0),cv2.LINE_AA)
                        flag1 = 1
                        
                    if (i==4 or i==7) and (704 < x < 921 and 360 < y < 486):
                        cv2.putText(image,'Good',(500,200),font,4,(0,255,0),cv2.LINE_AA)
                        flag2 = 1
                    
                    if (i==10 or i==13) and (124 < x < 622 and 785 < y < 949):
                        cv2.putText(image,'Good',(200,500),font,4,(0,0,255),cv2.LINE_AA)
                        flag5 = 1
                    
                    if i == 0:
                        head = y
                    if i == 8:
                        hip = y
                    if i == 4:
                        handr = x
                    if i == 10:
                        legr = x
                    if i == 7:
                        handl = x
                    if i == 13:
                        legl = x
                if hip < head:
                    cv2.putText(image,'Good',(800,200),font,4,(255,255,0),cv2.LINE_AA)
                    flag3 = 1
                    
                if (legr < handr or legr < handl or legl < handr or legl < handl) and handr < width/2 and handr != 0 and legr != 0 and handl != 0 and legl != 0:
                    cv2.putText(image,'Bad',(500,280),font,4,(255,0,255),cv2.LINE_AA)
                    flag4 = -1
                
                cv2.putText(img,str(handr) + "-" + str(legr),(0,50),font,2,(0,0,0),cv2.LINE_AA)
            else:
                for j in args.draw_point:
                    for i, (x, y, v) in enumerate(pose):
                        if(j == i or j < 0):
                            cv2.circle(image, (x, y), 5, Colors[i], -1)
                        
                        if (i==10 or i==13) and (1485 < x < 1782 and 695 < y < 845):
                            cv2.putText(image,'True',(1200,500),font,4,(255,0,0),cv2.LINE_AA)
                        if (i==4 or i==7) and (704 < x < 921 and 360 < y < 486):
                            cv2.putText(image,'True',(500,200),font,4,(0,255,0),cv2.LINE_AA)
                        if (i==10 or i==13) and (124 < x < 622 and 785 < y < 949):
                            cv2.putText(image,'True',(200,500),font,4,(0,0,255),cv2.LINE_AA)
                        
                        
                        #if(pose[10].x > 1425 and pose[10].x < 1640 and pose[10].y > 360 and pose[10].y < 486):
                        #    cv2.putText(img,'OK1',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
                        #if(pose[10].x > 124 and pose[10].x < 622 and pose[10].y > 785 and pose[10].y < 949):
                        #    cv2.putText(img,'OK2',(10,300), font, 4,(255,255,255),2,cv2.LINE_AA)
                    
                
            
            #評価を行う
'''     
'''
        # each person detected
        for person_pose in person_pose_array:
            unit_length = pose_detector.get_unit_length(person_pose)
    
            # face estimation
            print("Estimating face keypoints...")
            cropped_face_img, bbox = pose_detector.crop_face(img, person_pose, unit_length)
            if cropped_face_img is not None:
                face_keypoints = face_detector(cropped_face_img)
                res_img = draw_face_keypoints(res_img, face_keypoints, (bbox[0], bbox[1]))
                # cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
    
            # hands estimation
            print("Estimating hands keypoints...")
            hands = pose_detector.crop_hands(img, person_pose, unit_length)
            if hands["left"] is not None:
                hand_img = hands["left"]["img"]
                bbox = hands["left"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="left")
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                # cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
    
            if hands["right"] is not None:
                hand_img = hands["right"]["img"]
                bbox = hands["right"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="right")
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                # cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
'''
#        cv2.imshow("Test", res_image)
#        out.write(res_img)
        # print('Saving result into result.png...')
        # cv2.imwrite('result.png', res_img)
