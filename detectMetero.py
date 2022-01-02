import time
import os
import glob
import cv2
import numpy as np

def comp_b2(A,B):
# https://nyanpyou.hatenablog.com/entry/2020/03/20/132937
    gray_img1 = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
#グレースケールの比較で作成したimg1用のマスク(img1の方が明るい画素を示す)
    mask_img1 = np.where(gray_img1>gray_img2, 255, 0).astype(np.uint8)
#img2用のマスク(0と255を入れ替え)(img2の方が明るい画素を示す)
    mask_img2 = np.where(mask_img1==255, 0, 255).astype(np.uint8)

#作成したマスクを使って元画像から抜き出し
    masked_img1 = cv2.bitwise_and(A, A, mask=mask_img1)
    masked_img2 = cv2.bitwise_and(B, B, mask=mask_img2)

    img3 = masked_img1 + masked_img2
    return img3

def detectMove(thresh, img,tc,HL,HH):
    n = 0
    detect = False
    tMax=0
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):
        if len(contours[i]) > 0:
            tArea=cv2.contourArea(contours[i])
            if tArea > tc:
                if tArea>tMax:
                    tMax=tArea
                rect = contours[i]
                x, y, w, h = cv2.boundingRect(rect)
                if y>HL and (y+h)<HH:
                    detect = True
                    #cv2.rectangle(img, (x-w, y-h), (x + w*2, y + h*2), (0,0,255), 3)
    return detect,tMax

def disp(tgt,COMP):
        hist=[]
        frames=0
        d_frames=0
        cap = cv2.VideoCapture(tgt)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        HL,HH = 0, H*0.8
        prev=None
        avg=None
        tc, th = 5, 30
        #print(W,H)
        ret=True
        while ret:
            ret, img = cap.read()
            
            if ret:
                frames = frames + 1
                if frames>30:
                    break
                if COMP:
                    if prev is None:
                            #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        prev = img
                    else:
                        c_img = comp_b2(prev,img)
                        prev = c_img
                else:
                    c_img = None
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                if avg is None:
                    avg = gray.copy().astype("float")
    
                cv2.accumulateWeighted(gray, avg, 0.5)
                frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
                thresh = cv2.threshold(frameDelta, th, 255, cv2.THRESH_BINARY)[1]
                detect,area = detectMove(thresh,img,tc,HL,HH)
                hist.append(area)

            k=cv2.waitKey(1)&0xFF
            if k == ord('q'):
                break

        return hist,c_img
        
def main():
    path="Z:DATA/20211214"
    cDIR=path+'/COMP'
    if not(os.path.exists(cDIR)):
        os.mkdir(cDIR)

    if os.path.exists(path):
        temp=glob.glob(path+'/*avi')
        files = sorted( temp, key = lambda file: os.path.getctime(file))
        n=0
        for tgt in files:
            tmp=tgt.split('/')
            tmp=tmp[len(tmp)-1]
            tmp=tmp.split('.')
            tmp=tmp[0].split('\\')
            iFile=path+'/COMP/'+tmp[1]+'.jpg'  # 生成する比較明合成ファイル名
            if os.path.exists(iFile):
                COMP=False
            else:
                COMP=True
            hist,img = disp(tgt,COMP)     # 動体検知と比較明合成の処理
            ac, amax = 0,0
            for area in hist[1:]:   # 動体として検知した領域の面積が連続してゼロ以上のフレーム数をカウント
                ac = ac + 1 
                if area == 0:
                    if ac > amax:
                        amax = ac
                    ac =0
            if amax > 4:
                if not COMP:
                    img=cv2.imread(iFile)
                cv2.imshow('TITL',img)
                print(f'{tmp[1]}\t{amax}\t{hist}')
                cv2.imwrite(iFile,img)
            k=cv2.waitKey(1)&0xFF
            if k == ord('x'):
                break
        print('Done.')
        cv2.destroyAllWindows()
    else:
        print('Directory not exists.')
   
    
if __name__ == '__main__':
    main()
