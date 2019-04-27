#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
from PIL import Image
import matplotlib.pyplot as plt


def LJCL(Imgname,pot,JTZB,FKZB):     # 处理 框 与 箭头 的关系
    # img = Image.open("C:\\Users\\DELL\\Desktop\\P/667.jpg")
    img = Image.open(Imgname)
    # Pot = [190,73]   # 截取滑窗的左上角坐标  已知
    Pot = pot   # 截取滑窗的左上角坐标  已知
    # img = Image.open("C:\\Users\\DELL\\Desktop\\P/s667.jpg")  #小图像
    plt.imshow(img)
    print("图片大小：{}".format(img.size))
    # plt.show()

    # 小箭头坐标点
    # d2 = [[15,29],[25,29],[20,38]]
    d2 = JTZB
    # 判断箭头指向
    JT1 = []
    JT2 = []
    JT1.insert(len(JT1),d2[0])
    k = 0 # 计数
    for j in d2:
        if k>0:
            if abs(JT1[0][1]-j[1])<=3:
                JT1.insert(len(JT1), j)
            else:
                JT2.insert(len(JT1), j)
        k = k + 1
    if(len(JT1)>len(JT2)):
        print("箭头向下")
        FX = "下"
    else:
        print("箭头向上")
        FX = "上"

    # 画图的 X Y 轴点  无用
    X2 = []
    Y2 = []
    for i in d2:
        # X2.append(i[0])
        X2.append(i[0]+Pot[0])
        # Y2.append(i[1])
        Y2.append(i[1]+Pot[1])
    plt.scatter(X2,Y2,color='r',s=1)
    x2 = []
    y2 = []
    x2.append(int((X2[0]+X2[1])/2))
    x2.append(X2[2])
    y2.append(0)
    y2.append(img.size[1])
    # print(x2,y2)
    plt.plot(x2,y2)
    # plt.show()


    # 上下框的四角坐标
    # P1 = [[24,2],[390,2],[24,44],[390,44],[21,107],[388,107],[21,149],[387,149]]
    P1 = FKZB
    # 上下框的坐标
    KD1 = []
    KD2 = []

    #  分框坐标
    k = 0
    KD1.insert(len(KD1),P1[0])
    # 分 上下框的坐标
    for j in P1:
        if k>0:
            if abs(KD1[0][1]-j[1])<=int(img.size[1]/3) and abs(KD1[0][0]-j[0])<=img.size[0]:   # 默认同一个框的上下高度不超过30
                KD1.insert(len(JT1), j)
            else:
                KD2.insert(len(JT1), j)
        k = k + 1

    # 判断 上下框   字典打标签
    # YZ1 = tuple(KD1)
    # YZ2 = tuple(KD2)
    # print(YZ1)
    if KD1[0][1]<=int(img.size[1]/3):
        ZD = {"上框":KD1,"下框":KD2}
    else:
        ZD = {"上框": KD2, "下框": KD1}
    # print(ZD['上框'])

    #  方向
    if FX == "下":
        print("上框：{} --》 下框：{}".format(ZD['上框'],ZD['下框']))
    else:
        print( "下框：{} --》 上框：{}".format(ZD['上框'],ZD['下框']))

    # 圈出框的范围   暂时无用
    X3 = []
    Y3 = []
    for j in P1:
        X3.append(j[0])
        Y3.append(j[1])
    # plt.scatter(X3,Y3,color='r',s=2)
    # plt.show()
    plt.plot(X3,Y3)
    plt.show()


if __name__ == '__main__':
    Imgname  = "C:\\Users\\DELL\\Desktop\\P/667.jpg"
    pot = [190,73]   # 截取滑窗的左上角坐标  已知
    JTZB = [[15,29],[25,29],[20,38]]    # 箭头坐标
    FKZB = [[24,2],[390,2],[24,44],[390,44],[21,107],[388,107],[21,149],[387,149]]   #方框坐标
    LJCL(Imgname,pot,JTZB,FKZB)        # 进行逻辑处理
