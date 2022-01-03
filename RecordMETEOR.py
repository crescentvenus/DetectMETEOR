# Usage:
# python3 recordMETEOR.py

import cv2
import numpy as np
import time
import datetime
import os
import sys

cPATH = 'rtsp://xxx:yyy@aaa.bbb.ccc.ddd/live'
PATH = "/home/xxxxx/DATA"
if cPATH is None:
    print('Camera not selected:', opt)
    exit(0)

print('Camera:', cPATH, ' PATH=', PATH)

def key(k):
    global th, tc, track, reverse, disp
    if k == ord('2'):
        th = th - 1
    elif k == ord('3'):
        th = th + 1
    elif k == ord('4'):
        tc = tc - 1
    elif k == ord('5'):
        tc = tc + 1
    elif k == ord('t'):
        track = not track
    elif k == ord('d'):
        disp = not disp


def rename_file(filename, HEAD, detect_counts, f_frame):
    new_name = fname.replace(
        HEAD, HEAD+'_' + f'{detect_counts:03}'+'_'+f'{f_frame:03}'+'_')
    os.rename(fname, new_name)


def put_info(frame, th, tc, track):
    now = datetime.datetime.today()
    text = now.strftime("%m%d %H%M%S")+' No:'+str(frame) + \
        ' TH:'+str(th)+" SZ:"+str(tc) + " Rec:"+str(track)
    return text


fontFace = cv2.FONT_HERSHEY_SIMPLEX
track = False
disp = True
avg = None
writer = None
detect_counts = 0
red, blue, green, yellow = (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)
time_start = time.time()
t_frame, frame = 0, 0
log = PATH+'/metro.log'

TITLE = "ATOM2"
HEAD = 'ATOM2'

capture = cv2.VideoCapture(cPATH)
W = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
W2 = int(W/2)
H2 = int(H/2)
H85 = int(H2*0.8)
th, tc = 30, 5
x, y = 0, 0
info_color = blue
iPos1 = (30, H2-50)
iPos2 = (30, H - 50)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
print('Size:', W, H)
ret, img = capture.read()
org_img = cv2.resize(img, dsize=(W2, H2))
cv2.imshow(TITLE, org_img)
while(True):
    ret, img = capture.read()
    if ret:
        t_frame = t_frame + 1

        org_img = cv2.resize(img, dsize=(W2, H2))
        # org_img=img[0:int(H2*0.85),0:W2]
        gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
        org = org_img.copy()
        if avg is None:
            avg = gray.copy().astype("float")
            continue
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, th, 255, cv2.THRESH_BINARY)[1]

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detect = False
        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                if cv2.contourArea(contours[i]) > tc:
                    rect = contours[i]
                    x, y, w, h = cv2.boundingRect(rect)
                    if (y+h) < H2*0.85:
                        detect = True
                        if disp:
                            cv2.rectangle(org_img, (x-w, y-h),
                                          (x + w*2, y + h*2), red, 3)
        if detect:
            detect_counts = detect_counts + 1
            time_start = time.time()
            if writer is None and track:
                info_color = red
                detect_counts, t_frame = 0, 0
                now = datetime.datetime.today()
                date = now.strftime("%Y%m%d")
                cDIR = PATH+'/'+date
                if not(os.path.exists(cDIR)):
                    os.mkdir(cDIR)
                fname = cDIR+'/' + HEAD + now .strftime("%Y%m%d_%H%M%S")+".avi"
                writer = cv2.VideoWriter(fname, fourcc, 15, (W2, H2))
        if time.time() - time_start > 3:
            if writer is not None:
                writer.release()
                rename_file(fname, HEAD, detect_counts, t_frame)
                frame, t_frame = 0, 0
                writer = None
                info_color = blue
        if writer is not None:
            frame = frame+1
        if disp:
            text = f'{datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")} No:{frame} TH:{th} SZ:{tc} Rec:{track}'
            org_img = cv2.putText(org_img, text, iPos1,
                                  fontFace, 0.6, color=info_color)
            org_img = cv2.line(org_img, (0, H85), (W2, H85), red, 1)
            cv2.imshow(TITLE, org_img)

        if writer is not None:
            text = f'{datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")} No:{frame} TH:{th} SZ:{tc} Rec:{track}'
            org = cv2.putText(org, text, iPos2, fontFace,
                              0.6, color=info_color)
            writer.write(org)
    else:
        now = datetime.datetime.today()
        date = now.strftime("%Y%m%d_%H%M%S")
        print('reconnect:', TITLE, date)
        capture.release()
        avg = None
        capture = cv2.VideoCapture(cPATH)
        
    k = cv2.waitKey(1) & 0xFF
    key(k)
    if k == ord('q'):
        break

capture.release()
if writer is not None:
    writer.release()
    rename_file(fname, HEAD, detect_counts, t_frame)
cv2.destroyAllWindows()
