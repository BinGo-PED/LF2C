# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from scipy.cluster.vq import *
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import math
from scipy.cluster.vq import *
plt.rcParams['font.sans-serif'] = ['SimHei']
import cherrypy

class position_2img():
    # data_path='zed_hd_pic/' #数据路径
    data_path='test_data/'    #数据路径
    filename='2K_3.png' #照片文件名

    data_dict = {}  #属性文件字典
    img_list = []   #照片序列

    # width = 1280
    # height = 720

    width = 2208
    height = 1242

    img_full = np.zeros((height,width*2,3))
    img_left = np.zeros((height,width,3))
    img_right = np.zeros((height,width,3))
    rectify_right = np.zeros((height,width,3))
    rectify_left = np.zeros((height,width,3))
    disp = np.zeros((height,width))
    points = np.zeros((height,width,3))
    colors = np.zeros((height,width,3))
    precision_pic = np.zeros((height, width))  # 存储orb获取的点云

    # img_full = np.zeros((1242, 4416, 3))
    # img_left = np.zeros((1242, 2208, 3))
    # img_right = np.zeros((1242, 2208, 3))
    # rectify_right = np.zeros((1242, 2208, 3))
    # rectify_left = np.zeros((1242, 2208, 3))
    # disp = np.zeros((1242, 2208))
    # points = np.zeros((1242, 2208, 3))
    # colors = np.zeros((1242, 2208, 3))
    # precision_pic = np.zeros((1242, 2208))   #存储orb获取的点云

    #相机参数width*720
    # left_M = np.array([[701.30767, 0., 636.91397],[0., 701.71629, 368.17512],[0., 0., 1.]])   #通过标定计算
    # right_M = np.array([[ 703.20700, 0., 647.75295],[0., 703.50619, 378.61292],[0., 0., 1.]])  #通过标定计算
    # left_D = np.array([ -0.17130, 0.01537, 0.00045, 0.00013, 0.00000])   #通过标定计算
    # right_D = np.array([-0.16935, 0.02024, 0.00031, 0.00092, 0.00000])    #通过标定计算
    # om = np.array([0.00407, 0.00408, -0.00142])
    # R = cv.Rodrigues(om)[0]
    # T = np.array([-119.53552, -0.09909, 0.64940])
    # Q = np.zeros((4,4))

    # 相机参数2208*1242
    left_M = np.array([[1404.74849, 0., 1099.96693], [0., 1405.14158, 642.77126], [0., 0., 1.]])  # 通过标定计算
    right_M = np.array([[1400.18609, 0., 1121.37301], [0., 1400.80113, 659.37693], [0., 0., 1.]])  # 通过标定计算
    left_D = np.array([-0.16738, 0.01348, 0.00029, -0.00044, 0.00000])  # 通过标定计算
    right_D = np.array([-0.15688, -0.02414, -0.00088, -0.00018, 0.00000])  # 通过标定计算
    om = np.array([0.00146, 0.00478, -0.00147])
    R = cv.Rodrigues(om)[0]
    T = np.array([-119.80305, 0.09109, -0.92168])
    Q = np.zeros((4, 4))

    pic_cor = [0,0]

    # 相机程序获得
    # left_M = np.array([[700.0, 0., 640.0], [0., 700.0, 360.0], [0., 0., 1.]])  # 相机程序获得
    # right_M = np.array([[700.0, 0., 640.0], [0., 700.0, 360.0], [0., 0., 1.]])  # 相机程序获得
    # left_D = np.array([-0.16, 0., 0., 0., 0.00000])  # 相机程序获得
    # right_D = np.array([-0.16, 0., 0., 0., 0.00000])  # 相机程序获得
    # om = np.array([0., 0., 0.])
    # R = cv.Rodrigues(om)[0]     #旋转矩阵
    # T = np.array([-120., 0., 0.])
    # Q = np.zeros((4, 4))

    print('R', R)

    # @cherrypy.expose
    # def index(self):
    #     return "Hello world!"

    # @cherrypy.expose
    # def get_par(self, Xcor, Ycor):
    #     # print('X,Y:',Xcor, Ycor)
    #     pic_cor = (Xcor, Ycor)
    #     self.pic_cor = pic_cor

    #照片读取
    def read_img(self,filename):
        # filename = position_2img.filename
        filepath=position_2img.data_path
        # img = cv.imread(filepath+filename)
        img = cv.imread(filepath+filename)
        print(filepath+filename)
        # print(filepath+filename)
        print('img:',img.shape, img.ndim)
        img_info=img.shape
        print('origin img width=', img_info[1], ' height=', img_info[0],' channal=', img_info[2])
        # cv.namedWindow('origin img', cv.WINDOW_GUI_NORMAL)
        # cv.imshow('origin img', img)
        # cv.waitKey(1000)
        # cv.destroyAllWindows()
        position_2img.img_full = img

    #图像切割
    def cut_img(self, filename):
        img = position_2img.img_full

        # path = position_2img.data_path
        path = 'zed_hd_cut/'
        # print(img[0][0])
        # print(img[0][1])
        img_info = img.shape
        width = img_info[1]
        height = img_info[0]
        print('width,height:', width, height)
        left_img = img[0:height, 0:int(width/2)]   #纵轴、横轴
        right_img = img[0:height, int(width/2):width]
        # cv.namedWindow('img', cv.WINDOW_GUI_NORMAL)
        # cv.imshow('img', left_img)
        str1 = 'left' + filename
        str2 = 'right' + filename
        print(path + str1)
        cv.imwrite(path + str1, left_img)
        cv.imwrite(path + str2, right_img)
        position_2img.img_left = left_img
        position_2img.img_right = right_img
        # names = [str1,str2]
        # return(names)

    # 添加点击事件
    def OnMouseAction(e, x, y, f, p):
        if e == cv.EVENT_LBUTTONDOWN:
            print('点击位置：', 'x=', x, 'y=', y)
            Xcor = x
            Ycor = y
            # filename = position_2img.filename
            position_2img.one_pic(Ycor, Xcor,Ycor)

    #图像矫正
    def img_rectify(self,filename):
        # img_name = position_2img.cut_img(self)  #切割照片
        # print(img_name)
        path = position_2img.data_path
        # left_img =cv.imread(path+img_name[0])
        # right_img =cv.imread(path+img_name[1])  #读取切割后的照片

        left_img = position_2img.img_left
        right_img = position_2img.img_right

        # size = (2208, 1242)
        size = (position_2img.width,position_2img.height)
        # cv.namedWindow('left img')
        # cv.imshow('left img', left_img)
        # cv.namedWindow('right img')
        # cv.imshow('right img', right_img)
        # cv.waitKey(2000)
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(position_2img.left_M, position_2img.left_D,position_2img.right_M,position_2img.right_D, size, position_2img.R,position_2img.T)
        left_map1, left_map2 = cv.initUndistortRectifyMap(position_2img.left_M,position_2img.left_D, R1, P1, size, cv.CV_16SC2)
        right_map1, right_map2 = cv.initUndistortRectifyMap(position_2img.right_M, position_2img.right_D, R2, P2, size, cv.CV_16SC2)
        print('Q:', Q)
        position_2img.Q = Q
        left_rectify = cv.remap(left_img, left_map1, left_map2, cv.INTER_LINEAR)
        right_rectify = cv.remap(right_img, right_map1, right_map2, cv.INTER_LINEAR)

        # cv.namedWindow('recright img')
        # cv.imshow('recright img', right_rectify)
        # cv.waitKey(2000)
        str1 = 'rectify_left_'
        str2 = 'rectify_right_'
        cv.imwrite(path + str1 + filename, left_rectify)
        cv.imwrite(path + str2 + filename, right_rectify)

        position_2img.rectify_left = left_rectify
        position_2img.rectify_right = right_rectify
        # return(left_rectify, right_rectify)

    #手动点击获取位置
    def hand_pos(self,filename):
        Tx = position_2img.T[0]
        Cx = -position_2img.Q[0][3]
        Cy = -position_2img.Q[1][3]
        f = position_2img.Q[2][3]
        print(Tx, Cx, Cy, f)
        path = position_2img.data_path
        left_re = position_2img.rectify_left
        right_re = position_2img.rectify_right
        # cv.imshow('left_re', left_re)
        # cv.imshow('right_re', right_re)
        # plt.imshow(right_re)
        # pos1 = plt.ginput(2)
        # print(pos1)
        # pos2 = plt.ginput(5)
        # print(pos2)
        point_left = (800, 578)
        point_right = (773, 577)
        print(point_left, point_right)
        d = point_left[0] - point_right[0]
        X = Tx*(point_left[0] - Cx)/d
        Y = Tx*(point_left[1] - Cy)/d
        Z = -Tx*f/d
        print('X,Y,Z:',X,Y,Z,'实际距离为2.817m')
        pix_range = 50
        left_X_range = (point_left[0]-pix_range, point_left[0]+pix_range)
        left_Y_range = (point_left[1]-pix_range, point_left[1]+pix_range)
        right_X_range = (point_right[0]-pix_range, point_right[0]+pix_range)
        right_Y_range = (point_right[1]-pix_range, point_right[1]+pix_range)
        left_roi = left_re[left_Y_range[0]:left_Y_range[1], left_X_range[0]:left_X_range[1]]
        # left_roi = left_re[478:678,700:900]
        right_roi = right_re[right_Y_range[0]:right_Y_range[1], right_X_range[0]:right_X_range[1]]
        cv.namedWindow('left_roi',cv.WINDOW_GUI_NORMAL)
        cv.namedWindow('right_roi',cv.WINDOW_GUI_NORMAL)

        orb = cv.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(left_roi, None)

        cv.drawKeypoints(left_roi, kp1, left_roi, color=(0, 255, 0), flags=0)
        kp2, des2 = orb.detectAndCompute(right_roi, None)
        cv.drawKeypoints(right_roi, kp2, right_roi, color=(0, 255, 0), flags=0)
        cv.imshow('left_roi', left_roi)
        cv.imshow('right_roi', right_roi)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        #挑选一定像素范围内的同名点
        good_matches = []
        nums = len(matches)
        ex_max = 5
        for i in range(nums):
            trainIdx_i = matches[i].trainIdx
            queryIdx_i = matches[i].queryIdx
            if (abs(kp1[queryIdx_i].pt[1] - kp2[trainIdx_i].pt[1]) < 10 and abs(kp1[queryIdx_i].pt[0]-50)<ex_max and abs(kp1[queryIdx_i].pt[1]-50) < ex_max):
                good_matches.append(matches[i])
        good_nums = len(good_matches)
        print('good_matches:',good_nums)
        outImg = np.zeros((pix_range, pix_range * 2, 3))
        img3 = cv.drawMatches(left_roi, kp1, right_roi, kp2, matches[:], outImg, flags=2)
        cv.namedWindow('good_matches', cv.WINDOW_GUI_NORMAL)
        cv.imshow('good_matches',img3)

        hand_points = np.zeros((good_nums,8))
        for j in range(good_nums):
            trainIdx_j = good_matches[j].trainIdx
            queryIdx_j = good_matches[j].queryIdx
            hand_points[j][0] = kp1[queryIdx_j].pt[0]+(point_left[0]-pix_range)
            hand_points[j][1] = kp1[queryIdx_j].pt[1]+(point_left[1]-pix_range)
            hand_points[j][2] = kp2[trainIdx_j].pt[0]+(point_right[0]-pix_range)
            hand_points[j][3] = kp2[trainIdx_j].pt[1]+(point_right[1]-pix_range)
            hand_points[j][4] = good_matches[j].distance
            # 实际坐标计算
            d = hand_points[j][0] - hand_points[j][2]  # 视差
            hand_points[j][5] = -Tx * (hand_points[j][0] - Cx) / d  # X
            hand_points[j][6] = -Tx * (hand_points[j][1] - Cy) / d  # Y
            hand_points[j][7] = -f * Tx / d  # Z

        for i in range(good_nums):
            print('X,Y,Z:', hand_points[i][5], hand_points[i][6], hand_points[i][7])

    #点击单张照片
    def one_pic(self, Xcor, Ycor):
        print('Xcor=',Xcor,'Ycor=',Ycor)
        Tx = position_2img.T[0]
        Cx = -position_2img.Q[0][3]
        Cy = -position_2img.Q[1][3]
        f = position_2img.Q[2][3]
        print(Tx, Cx, Cy, f,'f*Tx=',f*Tx/1000)
        # hand_cor = [660,128]
        # hand_cor = [823,262]
        hand_cor = [Xcor,Ycor]
        path = position_2img.data_path
        left_re = position_2img.rectify_left
        right_re = position_2img.rectify_right
        # point_left = (800, 578)
        point_left = (hand_cor[0], hand_cor[1])
        point_right = point_left
        pix_range_left = 50
        pix_range_right = int(point_left[0]/10)
        left_X_range = (point_left[0] - pix_range_left, point_left[0] + pix_range_left)
        left_Y_range = (point_left[1] - pix_range_left, point_left[1] + pix_range_left)
        right_X_range = (point_right[0] - pix_range_right, point_right[0] + pix_range_left)
        right_Y_range = (point_right[1] - pix_range_left, point_right[1] + pix_range_left)

        left_roi = left_re[left_Y_range[0]:left_Y_range[1], left_X_range[0]:left_X_range[1]]
        right_roi = right_re[right_Y_range[0]:right_Y_range[1], right_X_range[0]:right_X_range[1]]
        orb = cv.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(left_roi, None)
        print('kp1:',len(kp1))
        cv.drawKeypoints(left_roi, kp1, left_roi, color=(0, 255, 0), flags=0)
        kp2, des2 = orb.detectAndCompute(right_roi, None)
        print('kp2:',len(kp2))
        cv.drawKeypoints(right_roi, kp2, right_roi, color=(0, 255, 0), flags=0)
        # cv.imshow('left_roi', left_re)
        # cv.imshow('right_roi', right_re)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        print(len(matches))
        outImg1 = np.zeros((pix_range_left, pix_range_right + pix_range_left, 3))
        img4 = cv.drawMatches(left_roi, kp1, right_roi, kp2, matches[:], outImg1, flags=2)
        cv.namedWindow('matches', cv.WINDOW_GUI_NORMAL)
        cv.imshow('matches', img4)

        good_matches = []
        ex_max = 5
        pix_range = 3
        nums = len(matches)
        print('nums', nums)
        for i in range(int(nums)):
            trainIdx_i = matches[i].trainIdx
            queryIdx_i = matches[i].queryIdx
            #匹配点筛选
            # good_matches.append(matches[i])
            print('左右Y差值:',abs(kp1[queryIdx_i].pt[1] - kp2[trainIdx_i].pt[1]))
            print('像素范围内:',abs(kp1[queryIdx_i].pt[0] - 50))
            print('像素范围内:',abs(kp1[queryIdx_i].pt[1] - 50))
            if (abs(kp1[queryIdx_i].pt[1] - kp2[trainIdx_i].pt[1]) < pix_range and abs(kp1[queryIdx_i].pt[0] - 50) < ex_max and abs(kp1[queryIdx_i].pt[1] - 50) < ex_max):
                good_matches.append(matches[i])

        good_nums = len(good_matches)
        print('good_nums:',good_nums)
        #
        #聚类筛选
        # good_points = np.array((good_matches,5))
        # good_matches2 = []
        # for t in range(good_nums):
        #     trainIdx_t = good_matches[t].trainIdx
        #     queryIdx_t = good_matches[t].queryIdx
        #     good_points[t][0] = kp2[trainIdx_t].pt[0]
        #     good_points[t][1] = kp2[trainIdx_t].pt[1]
        #     good_points[t][2] = trainIdx_t
        #     good_points[t][3] = queryIdx_t
        #     good_points[t][4] = t
        # print('goodpoints',good_points)

        outImg = np.zeros((pix_range_left, pix_range_right + pix_range_left, 3))
        img3 = cv.drawMatches(left_roi, kp1, right_roi, kp2, good_matches[:], outImg, flags=2)
        cv.namedWindow('good_matches', cv.WINDOW_GUI_NORMAL)
        cv.imshow('good_matches', img3)

        hand_points = np.zeros((good_nums, 8))
        for j in range(good_nums):
            trainIdx_j = good_matches[j].trainIdx
            queryIdx_j = good_matches[j].queryIdx
            hand_points[j][0] = kp1[queryIdx_j].pt[0] + (point_left[0] - pix_range_left)
            hand_points[j][1] = kp1[queryIdx_j].pt[1] + (point_left[1] - pix_range_left)
            hand_points[j][2] = kp2[trainIdx_j].pt[0] + (point_right[0] - pix_range_right)
            hand_points[j][3] = kp2[trainIdx_j].pt[1] + (point_right[1] - pix_range_left)
            hand_points[j][4] = good_matches[j].distance
            # 实际坐标计算
            d = hand_points[j][0] - hand_points[j][2]  # 视差
            hand_points[j][5] = -Tx * (hand_points[j][0] - Cx) / d  # X
            hand_points[j][6] = -Tx * (hand_points[j][1] - Cy) / d  # Y
            hand_points[j][7] = -f * Tx / d  # Z

        dis = []
        for i in range(good_nums):
            print('X,Y,Z:', hand_points[i][5], hand_points[i][6], hand_points[i][7])
            dis.append(math.sqrt(hand_points[i][5] * hand_points[i][5]+ hand_points[i][6] * hand_points[i][6]+ hand_points[i][7] * hand_points[i][7]))
        dis_array = np.array(dis)
        # xyz_all = hand_points[:, 5:8]
        # xyz = xyz_all.mean(axis=0)
        # print('xyz:', xyz)
        # print('xyz_sqrt:', math.sqrt(xyz[0] * xyz[0]+ xyz[1] * xyz[1]+ xyz[2] * xyz[2]))
        print('distance=',dis_array.mean()/1000)

def main():
    ex=position_2img()
    # cherrypy.quickstart(ex)
    filename=ex.filename
    print(filename)
    ex.read_img(filename)
    ex.cut_img(filename)
    ex.img_rectify(filename)
    # ex.pc_get(filename)
    left_rectify = ex.rectify_left
    cv.namedWindow('recleft img', cv.WINDOW_GUI_NORMAL)
    cv.imshow('recleft img', left_rectify)
    cv.setMouseCallback('recleft img', position_2img.OnMouseAction, None)

    #标定照片切割
    # pic_num = 10
    # for i in range(1,pic_num+1):
    #     filename = str(i)+'.jpg'
    #     print(filename)
    #     ex.read_img(filename)
    #     ex.cut_img(filename)

    cv.waitKey(0)
    cv.destroyAllWindows()
main()