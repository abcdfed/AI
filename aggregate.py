#coding=utf-8

import numpy as np
from pyramid.traversal import model_path
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
from configparser import *
from skimage import color
import os
import glob
import re
import datetime
from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdf2image import convert_from_path, convert_from_bytes
import tempfile
from pyramid.traversal import model_path
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
import cv2
from configparser import *
from skimage import color
import matplotlib.pyplot as plt

def split_pdf(infile, out_path):
    """
    将传进来的ｐｄｆ拆分成一页一页的ｐｄｆ，传给，turn_picture函数，
    :param infile: 待拆分的pdf文件
    :param out_path: 拆分成单页的pdf文件的存储路径
    """


    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(infile, 'rb') as infile:

        reader = PdfFileReader(infile)
        number_of_pages = reader.getNumPages()  #计算此PDF文件中的页数

        for i in range(number_of_pages):

            global number
            number = i
            writer = PdfFileWriter()
            writer.addPage(reader.getPage(i))
            temp = '/home/zheng/zheng/1pic/temp/'
            out_file_name = temp + 'asdf_asda'+'.pdf'



            with open(out_file_name, 'wb') as outfile:
                writer.write(outfile)
            yield i


def turn_picture(in_File, out_Path):
    '''
    将一个ｐｄｆ文件（该ｐｄｆ文件是单独一页的）转化成图片
    :param in_File: ｐｄｆ文件所在文件夹
    :param out_Path:
    :return:
    '''
    for i in split_pdf(in_File, out_Path):
        temp = '/home/zheng/zheng/1pic/temp/'
        out_file = temp + 'asdf_asda'+'.pdf'
        with tempfile.TemporaryDirectory() as path:
            images = convert_from_path(out_file)
            for index, img in enumerate(images):
                global number
                img.save('{}/page_{}_{}.png'.format(out_Path, number, 4_13_6))
                print('已经转化'+str(number))




# todo 去除四周留白
def wipeOff255(temp):
    """
    辅助用于图的检测
    去除四周留白，　并记录最上面的空白去了多少
    :param temp: numpy.array. 这个参数基本就是cv2.imread()读进来的
    :return: temp, number, number_bottom: 去完白的图片（数组，numpy.array),
            上下左右中的上去了多少, 上下左右中的下去了多少
    """

    number = 0
    number_bottom = 0
    stop = 1
    while stop and temp.shape[0] > 130:
        b = temp[:1, :]
        if b.sum() == temp.shape[1]*255:
            temp = temp[1:, :]
            number += 1
        else:
            stop = 0



    stop = 1
    while stop and temp.shape[0] > 130:
        b = temp[temp.shape[0]-1:, :]
        if b.sum() == temp.shape[1]*255:
            temp = temp[:temp.shape[0]-1, :]
            number_bottom += 1
        else:
            stop = 0

    temp = temp.T

    stop = 1
    while stop and temp.shape[0] > 130:
        b = temp[:1, :]
        if b.sum() == temp.shape[1]*255:
            temp = temp[1:, :]
        else:
            stop = 0

    stop = 1
    while stop and temp.shape[0] > 130:
        b = temp[temp.shape[0]-1:, :]
        if b.sum() == temp.shape[1]*255:
            temp = temp[:temp.shape[0]-1, :]
        else:
            stop = 0
    temp = temp.T
    return temp, number, number_bottom


def isPresistBy255(image):
    '''
    图的检测
    检查图片的每一行像素，　如果有一行全部为２５５
    ，既这一行是全白，那么返回Ｆａｌｓｅ
    一行也没有就返回Ｔｒｕｅ
    :param image: 要检查的图片
    :return:
    '''
    for i in range(image.shape[0]):
        temp = image[i: i+1, :]
        if temp.sum() == image.shape[1]*255:
            return False
    return True

def IOU(x1, h1, x2, h2):
    """
    图的检测
    计算交并比，返回结果
    :param x1:　矩阵，左上角的点的ｘ坐标
    :param h1:　（这个图的）高度
    :param x2:　另一矩阵的，左上角的ｘ坐标
    :param h2:　另一高度
    :return:
    """
    y1 = x1+h1
    y2 = x2+h2
    w = max(y1, y2) - min(x1, x2)
    if w > 0:
        maxOf4 = max(x1, y1, x2, y2)
        minOf4 = min(x1, y1, x2, y2)

        # print(maxOf4, minOf4)

        return w/(maxOf4 - minOf4)

    else:
        return -1


def integration(a, b, image, clf):
    '''
    图的检测
    将两个框合并，　按照最大的合并，尽可能的多
    :param a:　图片１
    :param b:　图片２
    :param image:　原始图片
    :param clf:　ｓｖｍ模型
    :return:
    '''

    x1 = a[2]
    h1 = a[0]
    x2 = b[2]
    h2 = b[0]

    y1 = x1+h1
    y2 = x2+h2

    x = min(x1, x2)
    y = max(y1, y2)
    # print(x1, y1, x2, y2, x, y)

    score = 1
    return (y-x, score, x)


# todo 改造的类似　，最大非非抑制值
def myNMS(detector, image, clf):
    """
    图的检测
    按线的大小排序，当有交并比大于０．５的将后面小的那个去掉
    detector的类型 [(), (), (), ()], 每一个元组都是(hOfpic, score, yOfpic)
    :param set: 一个ｓｅｔ（）
    :return: 返回一个操作完的ｌｉｓｔ，　ｌｉｓｔ的元素是一个二元数组
    """


    # print("------------------------------2-------------------------------------")

    detector.sort(reverse=True)

    # for i in detector:
        # print(i)

    # print("-----------------------------3---------------------------------------")


    for index, i in enumerate(detector):
        loc = detector.index(i) + 1
        while loc < len(detector):
            if IOU(detector[loc][2], detector[loc][0], i[2], i[0]) > 0.6:
                detector[index] = integration(detector[index], detector[loc], image, clf)
                del[detector[loc]]
            else:
                loc += 1



def verticalSidingWindows(PathOfpic,
                          PathOfPkl,
                          PathOfpic_outPut,
                          high=300,
                          ):
    '''
    垂直的滑窗，图的检测
    :param PathOfpic:　ｐｄｆ图片的地址
    :param PathOfPkl:　ｓｖｍ的模型的地址
    :param PathOfpic_outPut:　结果输出的文件夹
    :param high:　最小的框的高度。
    :return:
    '''

    print("start")
    image = cv2.imread(PathOfpic, 0)
    try:
        print(image.shape)
    except:
        print("没有这个图片")
        return
    # print(PathOfPkl)
    print('导入模型')
    clf = joblib.load(PathOfPkl)
    # 用于存放已经滑过的框，防止重复
    operated = set()
    # 用于存放最有要留下的框
    detector = []

    # 如果这一张全都是文字就排除
    isAllwords = 1
    startHigh = high

    while high < image.shape[0]:
        number = 0
        # 滑窗的左上角的坐标
        y = 0
        while y < image.shape[0] -high:

            yOfpic = y
            hOfpic = high

            temp = image[y:y+high]
            temp, number_qude, number_qude_buttom = wipeOff255(temp)

            yOfpic += number_qude
            hOfpic -= number_qude
            hOfpic -= number_qude_buttom

            item = str(yOfpic) + " " + str(hOfpic)
            if item not in operated:
                operated.add(item)
                # print("                            ", item, number_qude)
            else:
                print('重复一个', "                  ", item, number_qude)
                y += 25
                continue

            if not(isPresistBy255(temp)):
                y += 25
                print("从", y, "高度为：", high, number, '排除')
                continue
            isAllwords = 0
            temp = cv2.resize(temp, (700, 400))
            number += 1
            print("从", y, "start,高度为：", high, number)
            y += 25
            # 计算每个窗口的Hog特征
            fd = hog(temp,
                     orientations=9,  # number of bins
                     pixels_per_cell=(5, 5),   # pixel per cell
                     cells_per_block=(2, 2),   # cells per blcok
                     block_norm="L2",   # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}, optional
                     transform_sqrt=True,  # power law compression (also known as gamma correction)
                     feature_vector=True)  # flatten the final vectors

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)


            score  = clf.decision_function(fd)



            if score > 0.8:
                detector.append((hOfpic, score, yOfpic))
                print('入围的哟：：：：：：：：：：：：', number)
                cv2.imwrite('/home/zheng/Pictures/result/'+'H'+str(number)+'_'+str(score)+'.jpg', temp)
            if score > 0:
                cv2.imwrite('/home/zheng/Pictures/result/'+str(number)+'_'+str(score)+'.jpg', temp)

        if high == startHigh and isAllwords == 1:
            break
        high += 30


    myNMS(detector, image, clf)

    for temp_i, i in enumerate(detector):
        print(type(i))
        print(type(image))
        print(type(temp_i))
        image_temp = image[i[2]: i[2]+i[0]]
        image_temp, _, _ = wipeOff255(image_temp)
        cv2.imwrite('/home/zheng/Pictures/result/'+str(temp_i)+'.jpg', image_temp)



def sliding_window(image, step_size, window_size ):
    '''
    滑窗，用于箭头的检测，
    :param image: 要滑的图片
    :param step_size: 步长
    :param window_size: 要滑的窗的大小
    :return: 返回（ｘ，　ｙ，　ｉｍａｇｅ【】）　ｘ坐标，ｙ坐标，和图片
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def detector(pathOfpic,
             pathOfResult,
             pathOfPkl_arrow='arrow_30*75_rbf.pkl'
             ):
    '''
    专门用于箭头的检测，打开图片，读取模型，检测箭头，保存
    :param pathOfpic: 要检测的图片的路径
    :param pathOfPkl_arrow: ｓｖｍ模型的地址
    :param pathOfResult:
    :return:
    '''
    im = cv2.imread(pathOfpic, 0)

    min_wdw_sz = (30, 75)
    step_size = (5, 5)
    downscale = 1.2
    i = 0


    # 导入SVM模型
    clf = joblib.load(pathOfPkl_arrow)
    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # 在图像金字塔模型中对每个滑动窗口经行预测

    number_id = 0

    number_plies = 0

    for im_scaled_ in pyramid_gaussian(im,
                                       downscale=downscale,
                                       max_layer=1):

        number_plies += 1
        print('.................................  No.', number_plies)
        im_scaled = im_scaled_*255
        cd = []
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled,step_size, min_wdw_sz ):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue


            # 计算每个窗口的Hog特征
            fd = hog(im_window,
                     orientations=9,  # number of bins
                     pixels_per_cell=(5, 5),   # pixel per cell
                     cells_per_block=(3, 3),   # cells per blcok
                     block_norm="L2",   # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}, optional
                     transform_sqrt=True,  # power law compression (also known as gamma correction)
                     feature_vector=True)  # flatten the final vectors

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)


            score  = clf.decision_function(fd)


            if pred == 1:
                flag = 0
                if score > 1.2:
                    detections.append(
                        (
                         int(x*(downscale**scale)),
                         int(y*(downscale**scale)),
                         clf.decision_function(fd),  #样本点到超平面的距离
                         int(min_wdw_sz[0]*(downscale**scale)),
                         int(min_wdw_sz[1]*(downscale**scale))
                        )
                    )
                    print('入围的哟：：：：：：：：：：：：', number_id)
                    flag = 1
                if flag == 1:
                    cv2.imwrite('/home/zheng/Pictures2/result/'+'H'+str(number_id)+'_'+str(score)+'.jpg', im_window)
                else:
                    cv2.imwrite('/home/zheng/Pictures2/result/'+str(number_id)+'_'+str(score)+'.jpg', im_window)

            number_id += 1
            if number_id%1000 == 0:
                print(number_id)

        scale+=1

    clone = im.copy()

    # 画出矩形框 2553  9826 15639 15640 15643 16031
    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 255, 0), thickness=2)
    rects = np.array([[x, y, x + w, y + h] for (x, y,_, w, h) in detections])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)

    for (xA, yA, xB, yB) in pick:
        cnt = np.array([[xA, yA], [xB, yB], [xA, yB], [xB, yA]])
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.drawContours(clone, [box], 0, (0, 0, 255), 2)
        cnt1 = clone[yA:yB, xA:xB]  # y0:y1,x0:x1
        i+=1
        global pic_name
        cv2.imwrite("/home/zheng/Pictures2/result/"+pic_name+'-'+str(i)+".jpg", cnt1)
    plt.axis("off")
    plt.imshow(im)
    plt.title("Raw Detections before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()
