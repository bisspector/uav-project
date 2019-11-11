import numpy as np
import cv2
import time
import math
import os
import zipfile
import traceback
import json
from pymavlink import mavutil
from app import db, fs
from bson.objectid import ObjectId
from mimetypes import guess_type


def task6(taskId):
    try:
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "PROCESSING"}})

        # Reading
        img_str = fs.get(task["input_file"]).read()
        nparr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print('slicing colors...')
        # convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        row, column, channels = img.shape
        new_val = 210

        #####################BLUR########################################
        beps = 35
        hsv = cv2.GaussianBlur(hsv, (beps, beps), 0)
        #####################BLUR########################################

        print(hsv.shape[0], 'x', hsv.shape[1])
        used = []
        for i in range(0, hsv.shape[0]):
            used.append([])
            for j in range(0, hsv.shape[1]):
                used[i].append(False)

        def bfs(y, x, val, eps):
            q = [(y, x)]
            while len(q) > 0:
                x = q[0][1]
                y = q[0][0]
                used[y][x] = True
                hsv[y][x][0] = val
                if x + 1 < hsv.shape[1] and used[y][x + 1] == False and abs(float(hsv[y][x + 1][0]) - val) <= eps:
                    q.append((y, x + 1))
                    used[y][x + 1] = True
                if x - 1 >= 0 and used[y][x - 1] == False and abs(float(hsv[y][x - 1][0]) - val) <= eps:
                    q.append((y, x - 1))
                    used[y][x - 1] = True

                if y + 1 < hsv.shape[0] and used[y + 1][x] == False and abs(float(hsv[y + 1][x][0]) - val) <= eps:
                    q.append((y + 1, x))
                    used[y + 1][x] = True
                if y - 1 >= 0 and used[y - 1][x] == False and abs(float(hsv[y - 1][x][0]) - val) <= eps:
                    q.append((y - 1, x))
                    used[y - 1][x] = True
                del q[0]

        print('\nworking with hsv...')
        for i in range(0, hsv.shape[0]):
            for j in range(0, hsv.shape[1]):
                for g in range(1, 11):
                    if i == hsv.shape[0] // 100 * g * 10 and j == 0:
                        print(g * 10, f'%')
                if used[i][j] == False:
                    bfs(i, j, hsv[i][j][0], 2.25)

        mask = cv2.inRange(hsv, (0, 75, 0), (33, 255, 255))

        # slice the red
        imask = mask > 0
        red = np.zeros_like(img, np.uint8)
        red[imask] = img[imask]

        # slice the yellow
        mask = cv2.inRange(hsv, (30.00001, 75, 0), (35, 255, 255))

        imask = mask > 0
        yel = np.zeros_like(img, np.uint8)
        yel[imask] = img[imask]

        for i in range(0, red.shape[0]):
            for j in range(0, red.shape[1]):
                for g in range(1, 11):
                   if i == hsv.shape[0] // 100 * g * 10 and j == 0:
                       print(g * 10, f'%')
                if (i - 1 >= 0) and (yel[i][j].all() != 0 or red[i][j].all() != 0) and (yel[i - 1][j].all() == 0 and red[i - 1][j].all() == 0):
                    img.itemset((i, j, 0), 255)
                    img.itemset((i, j, 1), 255)
                    img.itemset((i, j, 2), 255)
                if (i + 1 < red.shape[0]) and (yel[i][j].all() != 0 or red[i][j].all() != 0) and (yel[i - 1][j].all() == 0 and red[i - 1][j].all() == 0):
                    img.itemset((i, j, 0), 255)
                    img.itemset((i, j, 1), 255)
                    img.itemset((i, j, 2), 255)

                if (j - 1 >= 0) and (yel[i][j].all() != 0 or red[i][j].all() != 0) and (yel[i][j - 1].all() == 0 and red[i][j - 1].all() == 0):
                    img.itemset((i, j, 0), 255)
                    img.itemset((i, j, 1), 255)
                    img.itemset((i, j, 2), 255)
                if (j + 1 < red.shape[1]) and (yel[i][j].all() != 0 or red[i][j].all() != 0) and (yel[i][j + 1].all() == 0 and red[i][j + 1].all() == 0):
                    img.itemset((i, j, 0), 255)
                    img.itemset((i, j, 1), 255)
                    img.itemset((i, j, 2), 255)

                if yel[i][j].all() != 0:
                    img.itemset((i, j, 1), new_val)
                if red[i][j].all() != 0:
                    img.itemset((i, j, 2), new_val)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("algo ended")

        # write output image into fs
        success, encoded_image = cv2.imencode('.png', img)
        imgdata = encoded_image.tobytes()
        content_type, _ = guess_type("test.png")
        fsfileid = fs.put(imgdata, content_type=content_type)

        # link file id with task document
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"output_file": fsfileid}})

        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "COMPLETED"}})
    except Exception:
        traceback.print_exc()
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "ERROR"}})


def task4(taskId):
    print("POINT 1")
    try:
        print("POINT 2")
        task = db.tasks.find_one_and_update(
            {"_id": taskId}, {"$set": {"state": "PROCESSING"}})
        print("POINT 3")
        inpzip = fs.get(task["input_file"])
        print("POINT 4")
        img = Merge.run(Merge(
            taskId, inpzip, task['etc']['alpha'], task['etc']['beta'], task['etc']['zoom']))
        print("POINT 5")
        success, encoded_image = cv2.imencode('.png', img)
        imgdata = encoded_image.tobytes()
        content_type, _ = guess_type("test.png")
        fsfileid = fs.put(imgdata, content_type=content_type)
        print("POINT 6")
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"output_file": fsfileid}})

        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "COMPLETED"}})

    except Exception:
        traceback.print_exc()
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "ERROR"}})


def task5(taskId):
    try:
        task = db.tasks.find_one_and_update(
            {"_id": taskId}, {"$set": {"state": "PROCESSING"}})

        inpzip = fs.get(task["input_file"])
        with zipfile.ZipFile(inpzip) as zip_ref:
            tdir = "/tmp/"+str(taskId)+"/"
            zip_ref.extractall(tdir)
            os.chdir(tdir)
            dirlist = os.listdir()
            im1 = open(tdir+dirlist[0])
            im2 = open(tdir+dirlist[1])

        img = fun(im1, im2, task['etc']['isField'])
        success, encoded_image = cv2.imencode('.png', img)
        imgdata = encoded_image.tobytes()
        content_type, _ = guess_type("test.png")
        fsfileid = fs.put(imgdata, content_type=content_type)
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"output_file": fsfileid}})

        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "COMPLETED"}})

    except Exception:
        traceback.print_exc()
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "ERROR"}})


def task2(taskId):
    try:
        task = db.tasks.find_one_and_update(
            {"_id": taskId}, {"$set": {"state": "PROCESSING"}})

        inpzip = fs.get(task["input_file"])
        with zipfile.ZipFile(inpzip) as zip_ref:
            tdir = "/tmp/"+str(taskId)+"/"
            zip_ref.extractall(tdir)
            os.chdir(tdir)
            dirlist = os.listdir()
        imlist = []
        for i in dirlist:
            imlist.append(tdir+i)

        img = fun2(imlist)[0][0]
        success, encoded_image = cv2.imencode('.png', img)
        imgdata = encoded_image.tobytes()
        content_type, _ = guess_type("test.png")
        fsfileid = fs.put(imgdata, content_type=content_type)
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"output_file": fsfileid}})

        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "COMPLETED"}})

    except Exception:
        traceback.print_exc()
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "ERROR"}})


def task3(taskId):
    print("POINT 1")
    try:
        print("POINT 2")
        task = db.tasks.find_one_and_update(
            {"_id": taskId}, {"$set": {"state": "PROCESSING"}})
        print("POINT 3")
        inpzip = fs.get(task["input_file"])
        print("POINT 4")
        img = Merge2.run(Merge2(
            taskId, inpzip, task['etc']['alpha'], task['etc']['beta'], task['etc']['zoom']))
        print("POINT 5")
        success, encoded_image = cv2.imencode('.png', img)
        imgdata = encoded_image.tobytes()
        content_type, _ = guess_type("test.png")
        fsfileid = fs.put(imgdata, content_type=content_type)
        print("POINT 6")
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"output_file": fsfileid}})

        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "COMPLETED"}})

    except Exception:
        traceback.print_exc()
        task = db.tasks.find_one_and_update(
            {"_id": ObjectId(taskId)}, {"$set": {"state": "ERROR"}})


'''
ALGOS
'''


def fun(before, after, fields=False):
    before = cv2.imread(before.name)
    after = cv2.imread(after.name)

    result_after = after.copy()

    ###############WORKING WITH IMAGES##############################################
    # ask = input('Do you want to work with field?[Y/n]: ')
    # if ask == 'Y' or ask == 'y':
    if fields == True:
        print('changintg from bgr to hsv...\n')
        before = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
        after = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)

        before[:, :, 2] = 120
        before[:, :, 1] = 120
        after[:, :, 2] = 120
        after[:, :, 1] = 120

        print('bluring...')
        # beps = 35
        beps = 35
        before = cv2.GaussianBlur(before, (beps, beps), 0)
        after = cv2.GaussianBlur(after, (beps, beps), 0)

        used = []
        for i in range(0, before.shape[0]):
            used.append([])
            for j in range(0, before.shape[1]):
                used[i].append(False)

        def zones(y, x, val, eps, hsv):
            nonlocal used
            q = [(y, x)]
            while len(q) > 0:
                x = q[0][1]
                y = q[0][0]
                used[y][x] = True
                # print(f'y: {y} x: {x} used[{y}][{x}]: {used[y][x]} val: {val}')
                hsv[y][x][0] = val
                if x + 1 < hsv.shape[1] and used[y][x + 1] == False and abs(float(hsv[y][x + 1][0]) - val) <= eps:
                    # print(abs(float(hsv[y][x + 1][0]) - val))
                    q.append((y, x + 1))
                    used[y][x + 1] = True
                if x - 1 >= 0 and used[y][x - 1] == False and abs(float(hsv[y][x - 1][0]) - val) <= eps:
                    # print(abs(float(hsv[y][x - 1][0]) - val))
                    q.append((y, x - 1))
                    used[y][x - 1] = True

                if y + 1 < hsv.shape[0] and used[y + 1][x] == False and abs(float(hsv[y + 1][x][0]) - val) <= eps:
                    # print(abs(float(hsv[y + 1][x][0]) - val))
                    q.append((y + 1, x))
                    used[y + 1][x] = True
                if y - 1 >= 0 and used[y - 1][x] == False and abs(float(hsv[y - 1][x][0]) - val) <= eps:
                    # print(abs(float(hsv[y - 1][x][0]) - val))
                    q.append((y - 1, x))
                    used[y - 1][x] = True
                del q[0]
            return hsv

        print('working with 1st photo...\nLoading...')
        for i in range(0, before.shape[0]):
            for j in range(0, before.shape[1]):
                for g in range(1, 11):
                    if i == before.shape[0] // 100 * g * 10 and j == 0:
                        print(g * 10, f'%')
                if used[i][j] == False:
                    # print(f"y: {i} - x: {j}")
                    before = zones(i, j, before[i][j][0], 15.25, before)

        used = []
        for i in range(0, before.shape[0]):
            used.append([])
            for j in range(0, before.shape[1]):
                used[i].append(False)

        print('working with 2nd photo...\nLoading...')
        for i in range(0, before.shape[0]):
            for j in range(0, before.shape[1]):
                for g in range(1, 11):
                    if i == before.shape[0] // 100 * g * 10 and j == 0:
                        print(g * 10, f'%')
                if used[i][j] == False:
                    # print(f"y: {i} - x: {j}")
                    after = zones(i, j, after[i][j][0], 15.25, after)

        print('changing from hsv to bgr...')
        before = cv2.cvtColor(before, cv2.COLOR_HSV2BGR)
        after = cv2.cvtColor(after, cv2.COLOR_HSV2BGR)

    # cv2.imwrite('before_blur.png', before)
    # cv2.imwrite('after_blur.png', after)
    # exit(code=0)
    ###############WORKING WITH IMAGES##############################################

    H = before.shape[0]
    W = before.shape[1]

    # W, H = before.shape[::-1]

    used = []
    maxSize = H * W - 1e5

    for i in range(0, H):
        used.append([])
        for j in range(0, W):
            used[i].append(False)

    def check(y1, x1, y2, x2):
        nonlocal before, after
        if before[y1][x1][0] != before[y2][x2][0] or before[y1][x1][1] != before[y2][x2][1] or before[y1][x1][2] != before[y2][x2][2]:
            return False
        return True

    def check2(y1, x1, y2, x2):
        nonlocal before, after
        if after[y1][x1][0] != after[y2][x2][0] or after[y1][x1][1] != after[y2][x2][1] or after[y1][x1][2] != after[y2][x2][2]:
            return False
        return True

    def bfs(y, x, maxSize):
        # print('bfs!')
        nonlocal H, W, before, after, used
        leftAns = 1e8
        rightAns = -1e8
        upperAns = 1e8
        bottomAns = -1e8
        col = 0

        q = []

        q.append((y, x))
        while(len(q) != 0):
            # used[q[0][0]][q[0][1]] = True
            # if col % 10000 == 0:
            #     print(col)
            leftAns = min(q[0][1], leftAns)
            rightAns = max(q[0][1], rightAns)
            upperAns = min(q[0][0], upperAns)
            bottomAns = max(q[0][0], bottomAns)

            if q[0][1] + 1 < W and check(q[0][0], q[0][1], q[0][0], q[0][1] + 1) == True and used[q[0][0]][q[0][1] + 1] == False:
                used[q[0][0]][q[0][1] + 1] = True
                q.append((q[0][0], q[0][1] + 1))
                col += 1
            if q[0][1] - 1 >= 0 and check(q[0][0], q[0][1], q[0][0], q[0][1] - 1) == True and used[q[0][0]][q[0][1] - 1] == False:
                used[q[0][0]][q[0][1] - 1] = True
                q.append((q[0][0], q[0][1] - 1))
                col += 1
            if q[0][0] + 1 < H and check(q[0][0], q[0][1], q[0][0] + 1, q[0][1]) == True and used[q[0][0] + 1][q[0][1]] == False:
                used[q[0][0] + 1][q[0][1]] = True
                q.append((q[0][0] + 1, q[0][1]))
                col += 1
            if q[0][0] - 1 >= 0 and check(q[0][0], q[0][1], q[0][0] - 1, q[0][1]) == True and used[q[0][0] - 1][q[0][1]] == False:
                used[q[0][0] - 1][q[0][1]] = True
                q.append((q[0][0] - 1, q[0][1]))
                col += 1

            q.pop(0)
        # print('End of loop')
        if col > maxSize:
            return (-1, -1, -1, -1, -1)
        return (leftAns, rightAns, upperAns, bottomAns, col)

    ###############################################################################

    def bfs2(y, x, maxSize):
        # print('bfs!')
        nonlocal H, W, before, after, used
        leftAns = 1e8
        rightAns = -1e8
        upperAns = 1e8
        bottomAns = -1e8
        col = 0

        q = []

        q.append((y, x))
        while(len(q) != 0):
            # used[q[0][0]][q[0][1]] = True
            # if col % 10000 == 0:
            #     print(col)
            leftAns = min(q[0][1], leftAns)
            rightAns = max(q[0][1], rightAns)
            upperAns = min(q[0][0], upperAns)
            bottomAns = max(q[0][0], bottomAns)

            if q[0][1] + 1 < W and check2(q[0][0], q[0][1], q[0][0], q[0][1] + 1) == True and used[q[0][0]][q[0][1] + 1] == False:
                used[q[0][0]][q[0][1] + 1] = True
                q.append((q[0][0], q[0][1] + 1))
                col += 1
            if q[0][1] - 1 >= 0 and check2(q[0][0], q[0][1], q[0][0], q[0][1] - 1) == True and used[q[0][0]][q[0][1] - 1] == False:
                used[q[0][0]][q[0][1] - 1] = True
                q.append((q[0][0], q[0][1] - 1))
                col += 1
            if q[0][0] + 1 < H and check2(q[0][0], q[0][1], q[0][0] + 1, q[0][1]) == True and used[q[0][0] + 1][q[0][1]] == False:
                used[q[0][0] + 1][q[0][1]] = True
                q.append((q[0][0] + 1, q[0][1]))
                col += 1
            if q[0][0] - 1 >= 0 and check2(q[0][0], q[0][1], q[0][0] - 1, q[0][1]) == True and used[q[0][0] - 1][q[0][1]] == False:
                used[q[0][0] - 1][q[0][1]] = True
                q.append((q[0][0] - 1, q[0][1]))
                col += 1

            q.pop(0)
        # print('End of loop')
        if col > maxSize:
            return (-1, -1, -1, -1, -1)
        return (leftAns, rightAns, upperAns, bottomAns, col)

    ###############################################################################

    print(f'{H} x {W}')

    print('loading...')
    for i in range(0, H):
        for j in range(0, W):
            for g in range(1, 11):
                if i == H // 100 * g * 10 and j == 0:
                    print(g * 10, f'%')
            leftAns, rightAns, upperAns, bottomAns, col = (-1, -1, -1, -1, -1)
            if used[i][j] == False:
                leftAns, rightAns, upperAns, bottomAns, col = bfs2(
                    i, j, maxSize)
            if col > 9 and abs(leftAns - rightAns) > 2 and abs(upperAns - bottomAns) > 2:
                sample = before[upperAns:bottomAns, leftAns:rightAns]
                # if i > 2:
                #     cv2.imwrite(f'{i}.png', sample)

                cv2.rectangle(result_after, (leftAns, upperAns),
                              (rightAns, bottomAns), (0, 255, 255), 2)

    used = []
    maxSize = H * W - 1e5

    for i in range(0, H):
        used.append([])
        for j in range(0, W):
            used[i].append(False)

    print('loading...')
    for i in range(0, H):
        for j in range(0, W):
            for g in range(1, 11):
                if i == H // 100 * g * 10 and j == 0:
                    print(g * 10, f'%')
            leftAns, rightAns, upperAns, bottomAns, col = (-1, -1, -1, -1, -1)
            if used[i][j] == False:
                leftAns, rightAns, upperAns, bottomAns, col = bfs(
                    i, j, maxSize)
            if col > 40 and abs(leftAns - rightAns) > 2 and abs(upperAns - bottomAns) > 2:
                sample = before[upperAns:bottomAns, leftAns:rightAns]
                # if i > 2:
                #     cv2.imwrite(f'{i}.png', sample)

                # cv2.rectangle(result_after, (leftAns, upperAns), (rightAns, bottomAns), (255, 0, 0), 1)
                # before = cv2.rectangle(before, (leftAns, upperAns), (rightAns, bottomAns), (255, 0, 0), 1)

                img_grey = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
                template = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

                w, h = template.shape[::-1]
                res = cv2.matchTemplate(
                    img_grey, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.99
                loc = np.where(res >= threshold)
                # print(f'len:', loc[::-1])

                for pt in zip(*loc[::-1]):
                    cv2.rectangle(result_after, pt,
                                  (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                    cv2.line(result_after, (abs(leftAns + rightAns) // 2, abs(upperAns +
                                                                              bottomAns) // 2), (pt[0] + w // 2, pt[1] + h // 2), (0, 0, 0))

                cv2.rectangle(result_after, (leftAns, upperAns),
                              (rightAns, bottomAns), (255, 0, 0), 2)
                if len(loc[0]) == 0:
                    cv2.rectangle(result_after, (leftAns, upperAns),
                                  (rightAns, bottomAns), (0, 0, 255), 2)

    # cv2.imwrite('rectangle.png', result_after)
    # cv2.imshow('after', after)
    # cv2.imshow('before', before)

    # cv2.waitKey(0)
    return result_after


def fun2(imgfiles):
    # global eps
    Max = 0
    show = []

    for imgway in imgfiles:
        img = cv2.imread(imgway)
        raz = 0
        for i in range(0, img.shape[0] - 1):
            for j in range(0, img.shape[1] - 1):
                raz += abs(int(img[i][j][0]) - int(img[i][j + 1][0]) + int(img[i][j][1]) - int(
                    img[i][j + 1][1]) + int(img[i][j][2]) - int(img[i][j + 1][2]))
                raz += abs(int(img[i][j][0]) - int(img[i + 1][j][0]) + int(img[i][j][1]) - int(
                    img[i + 1][j][1]) + int(img[i][j][2]) - int(img[i + 1][j][2]))
                # j+= eps
            # i+= eps
        raz = int(raz * 1e6 / (img.shape[0] * img.shape[1]))
        show.append((img, raz))
        print(raz)

    def sortSecond(val):
        return val[1]

    show.sort(key=sortSecond, reverse=True)

    return show


class Merge():
    CameraLogs = []
    Photos = []
    horizontalAngle = 90
    verticalAngle = 90
    zoom = 1

    def extract_array(self, log, param):
        print("EXTRACT_ARRAY")
        if log[-1] == '':
            log = log[:-1]
        res = []
        for i in range(0, len(log)):
            string = json.loads(log[i])
            # print(string['data'])
            res.append(string['data'])
        # print(res)
        # return res
        return res

    def __init__(self, taskId, inpzip, alpha, beta, zoom):
        self.horizontalAngle = alpha
        self.verticalAngle = beta
        self.zoom = zoom

        # unzip zipfile
        with zipfile.ZipFile(inpzip) as zip_ref:
            tdir = "/tmp/"+str(taskId)+"/"
            zip_ref.extractall(tdir)
            os.chdir(tdir)

        # get data from tlog
        dirlist = os.listdir()
        imagelist = sorted([file for file in dirlist if not (file.endswith(
            '.json') or file.endswith('.tlog'))])
        zf = zipfile.ZipFile(inpzip)
        # print("$$$$", zf.namelist())
        for i in range(0, len(imagelist)):
            print(imagelist[i][:2])
            if (imagelist[i][:2] == '__'):
                continue
            # if (imagelist[i][:2])
            data = zf.read(imagelist[i])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            self.Photos.append(img)
        for file in dirlist:
            if file.endswith('.json') or file.endswith('.tlog'):
                logfile = file

        if logfile.endswith(".tlog"):
            mlog = mavutil.mavlink_connection(logfile)
            while True:
                m = mlog.recv_match(type=["CAMERA_FEEDBACK"])
                if m is None:
                    break
                self.CameraLogs.append(m.to_dict())
        elif logfile.endswith(".json"):
            data = open('logs.json').read()
            data = data.split('\n')
            data = data[:len(data) - 1]
            # print(data)
            arr = self.extract_array(data, "CAMERA_FEEDBACK")
            print(arr)
            self.CameraLogs = arr

    def angleToMeters(self, lon, lat, helpLat):
        t = []
        t.append(lon * 111321 * math.cos(helpLat * math.pi / 180))
        t.append(lat * 111134)
        return t

    def spinDotAroundCenter(self, x0, y0, x, y, angle):
        x1 = x - x0
        y1 = y - y0
        x2 = x1 * math.cos(angle / 180 * math.pi) - y1 * \
            math.sin(angle / 180 * math.pi)
        y2 = x1 * math.sin(angle / 180 * math.pi) + y1 * \
            math.cos(angle / 180 * math.pi)
        x2 += x0
        y2 += y0
        ans = [x2, y2]
        return ans

    def overflow(self, inp, background, photo):  # Don't touch, ub'u nahuy
        if (abs(inp[0][0] - inp[3][0]) * abs(inp[0][1] - inp[3][1]) == 0):
            return background
        dots = []
        positions = []
        positions2 = []
        count = 0
        for i in range(0, 4):
            x = inp[i][0]
            y = inp[i][1]
            positions.append([x, y])
            if(count != 3):
                positions2.append([x, y])
            elif(count == 3):
                positions2.insert(2, [x, y])
            count += 1
        height, width = background.shape[:2]
        h1, w1 = photo.shape[:2]
        pts1 = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])
        pts2 = np.float32(positions)
        h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        height, width, channels = background.shape
        im1Reg = cv2.warpPerspective(photo, h, (width, height))
        mask2 = np.zeros(background.shape, dtype=np.uint8)
        roi_corners2 = np.int32(positions2)
        channel_count2 = background.shape[2]
        ignore_mask_color2 = (255,)*channel_count2
        cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
        mask2 = cv2.bitwise_not(mask2)
        masked_image2 = cv2.bitwise_and(background, mask2)
        final = cv2.bitwise_or(im1Reg, masked_image2)
        return final

    def getDots(self, x, y, alt, roll, pitch, yaw):  # Точки углов фотографии в пространстве
        photoHeightMeters = 2 * alt * \
            math.tan(self.verticalAngle / 360 * math.pi)
        photoWidthMeters = 2 * alt * \
            math.tan(self.horizontalAngle / 360 * math.pi)
        dots = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ]
        dots[1] = [x - photoWidthMeters / 2, y - photoHeightMeters / 2]
        dots[0] = [x + photoWidthMeters / 2, y - photoHeightMeters / 2]
        dots[3] = [x - photoWidthMeters / 2, y + photoHeightMeters / 2]
        dots[2] = [x + photoWidthMeters / 2, y + photoHeightMeters / 2]

        dots[0] = self.spinDotAroundCenter(x, y, dots[0][0], dots[0][1], yaw)
        dots[1] = self.spinDotAroundCenter(x, y, dots[1][0], dots[1][1], yaw)
        dots[2] = self.spinDotAroundCenter(x, y, dots[2][0], dots[2][1], yaw)
        dots[3] = self.spinDotAroundCenter(x, y, dots[3][0], dots[3][1], yaw)
        return dots

    def run(self):
        print("RUN")
        for e in self.CameraLogs:
            print(e)
        result = np.zeros((1, 1, 3), np.uint8)
        result[0:result.shape[0], 0:result.shape[1]] = (255, 255, 255)
        resMinX, resMinY = 1e9, 1e9
        resMaxX, resMaxY = -1e9, -1e9
        photosDots = []
        for i in range(0, len(self.CameraLogs)):
            info = self.CameraLogs[i]
            alt = abs(info['alt_rel']) * self.zoom
            x, y = self.angleToMeters(
                info['lng'] / 1e7, info['lat'] / 1e7, self.CameraLogs[0]['lat'] / 1e7)
            x *= self.zoom
            y *= self.zoom
            dots = self.getDots(
                x, y, info['alt_rel'] * self.zoom, info['roll'], info['pitch'], info['yaw'] * (-1) + 180)
            photosDots.append(dots)
            if (abs(self.CameraLogs[i]['roll']) > 10 or abs(self.CameraLogs[i]['pitch']) > 10):
                continue
            for j in range(0, 4):
                resMinX = min(resMinX, dots[j][0])
                resMaxX = max(resMaxX, dots[j][0])
                resMinY = min(resMinY, dots[j][1])
                resMaxY = max(resMaxY, dots[j][1])
        startX = resMinX
        startY = resMinY
        result = np.zeros((int(resMaxY - resMinY + 1),
                           int(resMaxX - resMinX + 1), 3), np.uint8)
        result[0:result.shape[0], 0:result.shape[1]] = (255, 255, 255)
        for i in range(0, len(self.CameraLogs)):
            if (abs(self.CameraLogs[i]['roll']) > 10 or abs(self.CameraLogs[i]['pitch']) > 10):
                continue
            print("OVERFLOWING: ", i)
            dots = photosDots[i]
            for j in range(0, 4):
                dots[j][0] = int(dots[j][0] - startX)
                dots[j][1] = int(resMaxY - dots[j][1])
            result = self.overflow(dots, result, self.Photos[i])
        return result


class Merge2():
    CameraLogs1 = []
    CameraLogs2 = []
    CameraLogs = []
    Photos1 = []
    Photos2 = []
    Photos = []
    horizontalAngle = 90
    verticalAngle = 70

    def extractArray(self, log, param):
        print("extractArray")
        if log[-1] == '':
            log = log[:-1]
        res = []
        for i in range(0, len(log)):
            string = json.loads(log[i])
            # print(string['data'])
            res.append(string['data'])
        # print(res)
        # return res
        return res

    def cmp(self, a):
        return a[0]

    def getQuality(self, img):
        img = cv2.resize(
            img, (int(img.shape[0] * 0.2), int(img.shape[1] * 0.2)))
        raz = 0
        for i in range(0, img.shape[0] - 1):
            for j in range(0, img.shape[1] - 1):
                # print(i, j)
                raz += abs(int(img[i][j][0]) - int(img[i][j + 1][0]) + int(img[i][j][1]) - int(
                    img[i][j + 1][1]) + int(img[i][j][2]) - int(img[i][j + 1][2]))
                # raz += abs(int(img[i][j][0]) - int(img[i + 1][j][0]) + int(img[i][j][1]) - int(
                #     img[i + 1][j][1]) + int(img[i][j][2]) - int(img[i + 1][j][2]))
        return int(raz)

    def __init__(self, taskId, inpzip, alpha, beta, zoom):
        self.horizontalAngle = alpha
        self.verticalAngle = beta
        self.zoom = zoom

        with zipfile.ZipFile(inpzip) as zip_ref:
            tdir = "/tmp/"+str(taskId)+"/"
            zip_ref.extractall(tdir)
            os.chdir(tdir)
            dirlist = [dir for dir in os.listdir() if not dir.startswith("__")]
            datasetd1 = tdir + dirlist[0] + "/"
            datasetd2 = tdir + dirlist[1] + "/"
            print(datasetd1)
            print(datasetd2)

            '''
            # ВОТ ЭТО ТЕБЕ НУЖНО
            '''
            paths_photos1 = sorted([datasetd1+file for file in os.listdir(datasetd1) if not (file.endswith(
                '.json') or file.endswith('.tlog') or file.startswith("__"))])
            paths_photos2 = sorted([datasetd2+file for file in os.listdir(datasetd2) if not (file.endswith(
                '.json') or file.endswith('.tlog') or file.startswith("__"))])
            paths_logs1 = sorted([datasetd1+file for file in os.listdir(datasetd1) if (file.endswith(
                '.json') or file.endswith('.tlog'))])
            paths_logs2 = sorted([datasetd2+file for file in os.listdir(datasetd2) if (file.endswith(
                '.json') or file.endswith('.tlog'))])





        
        print(")(*&^%$#@)", paths_logs1)
        # print(")(*&^%$#@)", paths_photos2)

        for i in range(0, len(paths_photos1)):
            # data = zf.read(paths_photos1[i])
            # img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            img = cv2.imread(paths_photos1[i])
            self.Photos1.append(img)
        
        for i in range(0, len(paths_photos2)):
            # data = zipfile.read(paths_photos2[i])
            # img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            img = cv2.imread(paths_photos1[i])
            self.Photos2.append(img)
        
        if paths_logs1[0].endswith(".tlog"):
            mlog = mavutil.mavlink_connection(paths_logs1[0])
            while True:
                m = mlog.recv_match(type=["CAMERA_FEEDBACK"])
                if m is None:
                    break
                self.CameraLogs1.append(m.to_dict())
        elif paths_logs1[0].endswith(".json"):
            data = open(paths_logs1[0]).read()
            data = data.split('\n')
            data = data[:len(data) - 1]
            # print(data)
            arr = self.extractArray(data, "CAMERA_FEEDBACK")
            print(arr)
            self.CameraLogs1 = arr.copy()


        if paths_logs2[0].endswith(".tlog"):
            mlog = mavutil.mavlink_connection(paths_logs2[0])
            while True:
                m = mlog.recv_match(type=["CAMERA_FEEDBACK"])
                if m is None:
                    break
                self.CameraLogs2.append(m.to_dict())
        elif paths_logs2[0].endswith(".json"):
            data = open(paths_logs2[0]).read()
            data = data.split('\n')
            data = data[:len(data) - 1]
            # print(data)
            arr = self.extractArray(data, "CAMERA_FEEDBACK")
            print(arr)
            self.CameraLogs2 = arr.copy()

        print("@@@@@@@@@@", self.CameraLogs1)
        print("@@@@@@@@@@", self.CameraLogs2)


        Qual1 = []
        Qual2 = []
        for i in range(0, len(self.Photos1)):
            print("QUALITY 1:", i)
            q = self.getQuality(self.Photos1[i])
            Qual1.append([])
            Qual1[i].append(q)
            Qual1[i].append(i)
        for i in range(0, len(self.Photos2)):
            print("QUALITY 2:", i)
            q = self.getQuality(self.Photos2[i])
            Qual2.append([])
            Qual2[i].append(q)
            Qual2[i].append(i)

        Qual1.sort(key=self.cmp)
        Qual2.sort(key=self.cmp)

        print("###########")
        print(len(Qual1), len(Qual2))
        print("###########")
        i, j = 0, 0
        while (i < len(Qual1) or j < len(Qual2)):
            if (i == len(Qual1)):
                self.CameraLogs.append(self.CameraLogs2[Qual2[j][1]])
                self.Photos.append(self.Photos2[Qual2[j][1]])
                j += 1
                continue
            if (j == len(Qual2)):
                self.CameraLogs.append(self.CameraLogs1[Qual1[i][1]])
                self.Photos.append(self.Photos1[Qual1[i][1]])
                i += 1
                continue

            if (Qual1[i][0] < Qual2[j][0]):
                self.CameraLogs.append(self.CameraLogs1[Qual1[i][1]])
                self.Photos.append(self.Photos1[Qual1[i][1]])
                i += 1
            else:
                self.CameraLogs.append(self.CameraLogs2[Qual2[j][1]])
                self.Photos.append(self.Photos2[Qual2[j][1]])
                j += 1

    def angleToMeters(self, lon, lat, helpLat):
        t = []
        t.append(lon * 111321 * math.cos(helpLat * math.pi / 180))
        t.append(lat * 111134)
        return t

    def spinDotAroundCenter(self, x0, y0, x, y, angle):
        x1 = x - x0
        y1 = y - y0
        x2 = x1 * math.cos(angle / 180 * math.pi) - y1 * \
            math.sin(angle / 180 * math.pi)
        y2 = x1 * math.sin(angle / 180 * math.pi) + y1 * \
            math.cos(angle / 180 * math.pi)
        x2 += x0
        y2 += y0
        ans = [x2, y2]
        return ans

    def overflow(self, inp, background, photo):  # Don't touch, ub'u
        if (abs(inp[0][0] - inp[3][0]) * abs(inp[0][1] - inp[3][1]) == 0):
            return background
        dots = []
        positions = []
        positions2 = []
        count = 0
        for i in range(0, 4):
            x = inp[i][0]
            y = inp[i][1]
            positions.append([x, y])
            if(count != 3):
                positions2.append([x, y])
            elif(count == 3):
                positions2.insert(2, [x, y])
            count += 1
        height, width = background.shape[:2]
        h1, w1 = photo.shape[:2]
        pts1 = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])
        pts2 = np.float32(positions)
        h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        height, width, channels = background.shape
        im1Reg = cv2.warpPerspective(photo, h, (width, height))
        mask2 = np.zeros(background.shape, dtype=np.uint8)
        roi_corners2 = np.int32(positions2)
        channel_count2 = background.shape[2]
        ignore_mask_color2 = (255,)*channel_count2
        cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
        mask2 = cv2.bitwise_not(mask2)
        masked_image2 = cv2.bitwise_and(background, mask2)
        final = cv2.bitwise_or(im1Reg, masked_image2)
        return final

    def getDots(self, x, y, alt, roll, pitch, yaw):  # Точки углов фотографии в пространстве
        photoHeightMeters = 2 * alt * \
            math.tan(self.verticalAngle / 360 * math.pi)
        photoWidthMeters = 2 * alt * \
            math.tan(self.horizontalAngle / 360 * math.pi)
        dots = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ]
        dots[1] = [x - photoWidthMeters / 2, y - photoHeightMeters / 2]
        dots[0] = [x + photoWidthMeters / 2, y - photoHeightMeters / 2]
        dots[3] = [x - photoWidthMeters / 2, y + photoHeightMeters / 2]
        dots[2] = [x + photoWidthMeters / 2, y + photoHeightMeters / 2]

        dots[0] = self.spinDotAroundCenter(x, y, dots[0][0], dots[0][1], yaw)
        dots[1] = self.spinDotAroundCenter(x, y, dots[1][0], dots[1][1], yaw)
        dots[2] = self.spinDotAroundCenter(x, y, dots[2][0], dots[2][1], yaw)
        dots[3] = self.spinDotAroundCenter(x, y, dots[3][0], dots[3][1], yaw)
        return dots

    def run(self):
        # print(dots)
        result = np.zeros((1, 1, 3), np.uint8)
        result[0:result.shape[0], 0:result.shape[1]] = (255, 255, 255)
        # return result

        # print(result)
        # return result
        startLon, startLat = self.CameraLogs[0]['lng'] / \
            1e7, self.CameraLogs[0]['lat'] / 1e7
        resMinX, resMinY = 1e9, 1e9
        resMaxX, resMaxY = -1e9, -1e9
        photosDots = []
        for i in range(0, len(self.CameraLogs)):
            info = self.CameraLogs[i]
            alt = abs(info['alt_rel']) * self.zoom
            x, y = self.angleToMeters(info['lng'] / 1e7, info['lat'] / 1e7, startLat)
            print("COUNTING: ", i, info['roll'], info['pitch'])
            x *= self.zoom
            y *= self.zoom
            dots = self.getDots(
                x, y, info['alt_rel'] * self.zoom, info['roll'], info['pitch'], (info['yaw']) * (-1) + 180)
            photosDots.append(dots)
            # print(dots)
            if (abs(self.CameraLogs[i]['roll']) > 10 or abs(self.CameraLogs[i]['pitch']) > 10):
                continue
            for j in range(0, 4):
                resMinX = min(resMinX, dots[j][0])
                resMaxX = max(resMaxX, dots[j][0])
                resMinY = min(resMinY, dots[j][1])
                resMaxY = max(resMaxY, dots[j][1])
        startX = resMinX
        startY = resMinY
        print("---", int(resMaxY - resMinY + 1), int(resMaxX - resMinX + 1))
        result = np.zeros((int(resMaxY - resMinY + 1),
                           int(resMaxX - resMinX + 1), 3), np.uint8)
        result[0:result.shape[0], 0:result.shape[1]] = (255, 255, 255)
        print("!!!!!!!!!!!!!!!!!!!!")
        for i in range(0, len(self.CameraLogs)):
            if (abs(self.CameraLogs[i]['roll']) > 10 or abs(self.CameraLogs[i]['pitch']) > 10):
                continue
            print("OVERFLOWING: ", i)
            dots = photosDots[i]
            for j in range(0, 4):
                dots[j][0] = int(dots[j][0] - startX)
                dots[j][1] = int(resMaxY - dots[j][1])
            result = self.overflow(dots, result, self.Photos[i])
        return result
