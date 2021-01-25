#!/usr/bin/env python
#coding=gbk
"""
@version: 1.0
@author: liaoliwei
@contact: levio@pku.edu.cn
@file: people_flow.py
@time: 2018/7/9 14:52
"""


import colorsys
import time
import sqlite3
from timeit import default_timer as timer
import requests
import json
import asyncio
import websocket
from datetime import datetime
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model
gpu_num=1
class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def get_state(self):
        mlist = [(1608521400,1608525000,"海茵",4),(1608528600,1608532200,"海茵",5),(1608620400,1608624000,"海茵",6),
                 (1608690600,1608694200,"海茵",8),(1608796800,1608800400,"海茵",6)]
        start_time = 0
        end_time =0
        ord_time = (start_time,end_time)
        ord_people = ""
        ord_number = 0
        for i in range(len(mlist)):
            if mlist[i][0]<time.time()<mlist[i][1]:
                start_time = mlist[i][0]
                end_time = mlist[i][1]
                ord_time = (start_time, end_time)
                ord_people=mlist[i][2]
                ord_number=mlist[i][3]

        return ord_time, ord_people ,ord_number

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        lbox = []
        lscore = []
        lclass = []
        for i in range(len(out_classes)):
            if out_classes[i] == 0:
                lbox.append(out_boxes[i])
                lscore.append(out_scores[i])
                lclass.append(out_classes[i])
        out_boxes = np.array(lbox)
        out_scores = np.array(lscore)
        out_classes = np.array(lclass)
        print('画面中有{}个人'.format(len(out_boxes)))
        meetingtime = time.time()
        #tup = (people, start_time,end_time,meeting_time,situation)
        uploadnow(out_boxes,meetingtime)
        tup = (len(out_boxes), meetingtime)
        conn = sqlite3.connect("meetingroom.db")
        c = conn.cursor()
        c.execute("insert into meetingroom VALUES(?,?)", tup)
        conn.commit()
        c.close()
        conn.close()
        if len(out_boxes) == 0:
            avg()



        """font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        font_cn = ImageFont.truetype(font='font/asl.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            show_str = ' 2画面中有'+str(len(out_boxes))+'个人  '
            # 填入数据库
            

            label_size1 = draw.textsize(show_str, font_cn)
            #print(label_size1)
            draw.rectangle(
                [10, 10, 10 + label_size1[0], 10 + label_size1[1]],
                fill=(255,255,0))
            draw.text((10,10),show_str,fill=(0, 0, 0), font=font_cn)

            del draw

        end = timer()"""
        #print(end - start)
        return image

    def close_session(self):
        self.sess.close()



def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()

        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        #cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

def upload_meetingroom():
    conn = sqlite3.connect("meetingroom.db")
    c = conn.cursor()
    c.execute("SELECT * from meetingroom")
    cursor = c.fetchall()
    for row in cursor:
        if type(row[0])==int:
            ws = websocket.create_connection("ws://47.89.240.122:2346?serial=100000001c273adb")
            ws.send(json.dumps(
                {"route": "/meetingroom/identifies", "person_num": row[0], "identify_time": row[1]}))
    conn.commit()
    c.close()
    conn.close()



def uploadnow(out_boxes,nowtime):
    try:
        ws = websocket.create_connection("ws://47.89.240.122:2346?serial=100000001c273adb")
        ws.send(json.dumps(
            {"route": "/meetingroom/identifies", "person_num": len(out_boxes), "identify_time": nowtime}))
        result = ws.recv()
        if json.loads(result)["status"]== True:
            print("实时数据上传")
    except OSError or ConnectionRefusedError:
        tup = (int(len(out_boxes)), nowtime)
        conn = sqlite3.connect("meetingroom.db")
        c = conn.cursor()
        c.execute("insert into meetingroom VALUES(?,?)", tup)
        conn.commit()
        c.close()
        conn.close()
        print("网络错误,存入数据库")

def avg():
    conn = sqlite3.connect("meetingroom.db")
    c = conn.cursor()
    c.execute("SELECT * from meetingroom")
    cursor = c.fetchall()
    number = []
    times = []
    for row in cursor:
        if type(row[0]) == int:
            number.append(row[0])
            times.append(row[1])
    avg = np.mean(number)
    if avg!=0:
    # print(avg)
        data = json.dumps(
            {"route": "/meetingroom/period", "avg_person": avg, "start_time": times[0], "end_time": times[-1]})
        ws = websocket.create_connection("ws://47.89.240.122:2346?serial=100000001c273adb")
        ws.send(data)
        result = ws.recv()
        if json.loads(result)["status"] == True:
            print("平均值上传成功")
        conn = sqlite3.connect("meetingroom.db")
        c = conn.cursor()
        c.execute("delete from meetingroom")
        conn.commit()
        c.close()
        conn.close()








if __name__ == '__main__':
    detect_img(YOLO())
