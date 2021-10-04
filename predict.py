

import keras

import cv2

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model, model_from_json
import csv
import pytesseract
import argparse

import locality_aware_nms as nms_locality
import lanms
from model import *
from losses import *
from data_processor import restore_rectangle


## Pre processing

### Helper functions

def resize_image(im, max_side_len=2400):
    
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_path", type=str, required=True,
    help="path to input text detector")
ap.add_argument("-v", "--video_name", type=str, required = True,
    help="path to input video file")
ap.add_argument("-o", "--output_dir", type=float, required = True,
    help="What is the output directory")
ap.add_argument("-p", "--padding", type=float, default=0.10,
    help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

gpu_list='0'

json_file = open(os.path.join('/'.join(model_path.split('/')[0:-1]), 'model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': RESIZE_FACTOR})
model.load_weights(model_path)

video_capture = cv2.VideoCapture(video_name)
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
frame_num = 0

res_file = os.path.join(
        output_dir,
        '{}.csv'.format(
            os.path.basename(video_name).split('.')[0]))

with open(res_file, 'w') as output:

  rows = []
  csvwriter = csv.writer(output) 
  csvwriter.writerow(["time","text"])
  while video_capture.isOpened():
      t = frame_num/fps
      x = '{}:{}'.format(t//60, t%60)
      row = [x]
      start = time.time()
      timer = {'net': 0, 'restore': 0, 'nms': 0}
      video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
      success, img = video_capture.read()

      if not success: break

      frame_num += 2*fps

      img_resized, (ratio_h, ratio_w) = resize_image(img)

      img_resized = (img_resized / 127.5) - 1
      
      score_map, geo_map = model.predict(img_resized[np.newaxis, :, :, :])

      timer['net'] = time.time() - start

      boxes, timer = detect(score_map=score_map, geo_map=geo_map, timer=timer)

      if boxes == []:
        continue
      if boxes is not None:
          boxes = boxes[:, :8].reshape((-1, 4, 2))
          boxes[:, :, 0] /= ratio_w
          boxes[:, :, 1] /= ratio_h

      for i in range(boxes.shape[0]):
        boxes[i] = sort_poly(boxes[i].astype(np.int32))

      b = boxes.astype(np.int16).reshape((-1,8))
      b = b[b[:,1].argsort()]
      temp = np.zeros(b.shape[0])
      temp[0] = b[0,1]
      for i in range(1,b.shape[0]):
        if b[i, 1] - b[i-1,1] <= 6: temp[i] = temp[i-1]
        else: temp[i] = b[i, 1]
      b[:,1]=temp

      a = b[b[:,0].argsort()]
      boxes = a[a[:,1].argsort()]

      all_text = ""        
      y = boxes[0,1]
      for box in boxes:
          if np.linalg.norm(box[0] - box[2]) < 5 or np.linalg.norm(box[3]-box[5]) < 5:
              continue
          
          startX = min(box[0], box[6])
          endX = max(box[2], box[4])
          startY = min(box[1], box[3])
          endY = max(box[5], box[7])

          y_pad = int((endY - startY) * padding)
          x_pad = int((endX - startX) * padding)
          roi = img[startY - y_pad:endY + y_pad, startX - x_pad:endX + x_pad]
          
          try: gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
          except:
            break
          gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
          gray = cv2.bitwise_not(img_bin)
          kernel = np.ones((2, 1), np.uint8)
          roi = cv2.erode(gray, kernel, iterations=1)
          roi = cv2.dilate(roi, kernel, iterations=1)

          if np.count_nonzero(roi)/(roi.shape[0]*roi.shape[1]) < 0.3: roi = 255 - roi

          text = pytesseract.image_to_string(roi, lang='eng', config='--psm 7')
          
          all_text += text.strip()
          all_text += ' '

          if box[1] != y: all_text += '\n'
          y = box[1]

      for box in boxes:
        img = cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

      clear_output(wait = True)
      row.append(all_text.replace('\n', " "))
      csvwriter.writerow(row)