from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import glob as glob
import shutil
import os
import cv2
import time
import math

def main(img_back, img_front, left, top):
    img = eval("putSprite_npwhere")(img_back, img_front, (left, top))
    return img

def putSprite_npwhere(back, front, pos):
    x, y = pos
    fh, fw = front.shape[:2]
    bh, bw = back.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + fw, bw), min(y + fh, bh)
    if not ((-fw < x < bw) and (-fh < y < bh)):
        return back
    front3 = front[:, :, :3]
    front_roi = front3[y1-y:y2-y, x1-x:x2-x]
    roi = back[y1:y2, x1:x2]
    tmp = np.where(front_roi==(0,0,0), roi, front_roi)
    back[y1:y2, x1:x2] = tmp
    return back

def putSprite_mask(back, front, pos):
    x, y = pos
    fh, fw = front.shape[:2]
    bh, bw = back.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + fw, bw), min(y + fh, bh)
    if not ((-fw < x < bw) and (-fh < y < bh)):
        return back
    front3 = front[:, :, :3]
    mask1 = front[:, :, :3]
    mask3 = 255 - cv2.merge((mask1, mask1, mask1))
    mask_roi = mask3[y1-y:y2-y, x1-x:x2-x]
    front_roi = front3[y1-y:y2-y, x1-x:x2-x]
    roi = back[y1:y2, x1:x2]
    tmp = cv2.bitwise_and(roi, mask_roi)
    tmp = cv2.bitwise_or(tmp, front_roi)
    back[y1:y2, x1:x2] = tmp
    return back

def iou(a, b):
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mn = max(ay_mn, by_mn)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    area = w * h
    iou = area / (a_area + b_area - area)
    return iou

def radian(img):
    angle = np.random.choice(np.arange(361), 1, replace=True)[0]
    angle_rad = np.deg2rad(angle)
    h, w = img.shape[:2]
    size = (w, h)
    w_rot = int(np.round(h * np.absolute(np.sin(angle_rad)) + w * np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h * np.absolute(np.cos(angle_rad)) + w * np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)
    center = (w / 2, h / 2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] -w / 2 + w_rot / 2
    affine_matrix[1][2] = affine_matrix[1][2] -h / 2 + h_rot / 2
    img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
    return img_rot

file = 'source/background.png'
background = cv2.imread(file, -1)

filepaths = glob.glob('source/*.jpg')

agents = dict()
dic = dict()
arr = []
for k, filepath in enumerate(filepaths):
    base = os.path.basename(filepath).split(".png")[0]
    agents[base] = cv2.imread(filepath, -1)
    dic[base] = k
    arr.append(base)

"""
num : image count
iters : 
"""
num = 5000
iters = 40
for i in range(num):

    h = background.copy()
    p = np.random.choice(arr, iters, replace=True)
    bboxs = []

    for j in range(iters):
        img = agents[p[j]]
        k = np.random.choice(np.arange(100), 1, replace=True)
        if k < 33:
            c = np.random.choice(np.arange(50, 110)/100, 1, replace=True)
            img = cv2.resize(img, dsize=None, fx=c[0], fy=c[0])
        k = np.random.choice(np.arange(100), 1, replace=True)

        if k < 33:
            s = np.random.choice(np.arange(50, 100)/100, 1, replace=True)
            v = np.random.choice(np.arange(50, 100)/100, 1, replace=True)
            img_hsy = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            s_mag = s
            v_mag = v
            img_hsy[:, :, 1] = img_hsy[:, :, 1] * s_mag
            img_hsy[:, :, 2] = img_hsy[:, :, 2] * v_mag
            img = cv2.cvtColor(img_hsy, cv2.COLOR_HSV2BGR)
        k = np.random.choice(np.arange(100), 1, replace=True)
        
        if k < 51:
            img = radian(img)
        
        n = 0
        while True:
            n += 1
            top = np.random.choice(np.arange(0, h.shape[0]), 1, replace=True)
            left = np.random.choice(np.arange(0, h.shape[1]), 1, replace=True)
            bottom = top + img.shape[0]
            right = left + img.shape[1]
            if bottom[0] < h.shape[0] and right[0] < h.shape[1]:
                flag = False
                for bbox in bboxs:
                    if iou(bbox[:4], [left[0], top[0], right[0], bottom[0]]) > 0.1:
                        flag = True
                        break

                if flag == False:
                    break
            
            if n == 100:
                break
        
        if flag == False:
            h = main(h, img, left[0], top[0])
            bboxs.append([left[0], top[0], right[0], bottom[0], p[j]])
    
    cv2.imwrite('result/images/' + str(i) + '.png', h)
    df = pd.DataFrame(bboxs)
    df.columns = ["xmin", "ymin", "xmax", "ymax", "label"]
    df["xmin"] = df["xmin"] / h.shape[1]
    df["ymin"] = df["ymin"] / h.shape[0]
    df["xmax"] = df["xmax"] / h.shape[1]
    df["ymax"] = df["ymax"] / h.shape[0]
    df["xcenter"] = (df["xmin"] + df["xmax"]) / 2
    df["ycenter"] = (df["ymin"] + df["ymax"]) / 2
    df["width"] = df["xmax"] - df["xmin"]
    df["height"] = df["ymax"] - df["ymin"]
    df["label_number"] = [dic[i] for i in df["label"]]
    df = df[["label_number", "xcenter", "ycenter", "width", "height"]]
    txt = "result/labels/" + str(i) + ".txt"
    df.to_csv(txt, header=None, index=None, sep=" ")

    print(str(i) + ".png done")