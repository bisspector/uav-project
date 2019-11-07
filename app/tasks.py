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
                # for g in range(1, 11):
                #    if i == hsv.shape[0] // 100 * g * 10 and j == 0:
                #        print(g * 10, f'%')
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    try:
        task = db.tasks.find_one_and_update(
            {"_id": taskId}, {"$set": {"state": "PROCESSING"}})

        inpzip = fs.get(task["input_file"])
        img = Merge.run(Merge(taskId, inpzip, task['etc']['alpha'], task['etc']['beta'], task['etc']['zoom']))

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


'''
ALGOS
'''

class Merge():
    CameraLogs = []
    Photos = []
    horizontalAngle = 90
    verticalAngle = 90
    zoom = 1

    def extract_array(self, log, param):
        if log[-1] == '':
            log = log[:-1]
        log = [json.loads(record) for record in log]
        desired = []
        for record in log:
            if record['meta']['type'] == param:
                desired.append(record)
        return desired


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
            data = str(data)[2:]
            data = data.split('{"meta": ')
            for i in range(0, len(data)):
                data[i] = data[i][:len(data[i]) - 2]
                data[i] = '{"meta": ' + data[i]
            data[len(data) - 1] = data[len(data) - 1][:len(data[len(data) - 1]) - 1]
            data = data[1:]
            arr = self.extract_array(data, "CAMERA_FEEDBACK")
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
        h, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
        height, width, channels = background.shape
        im1Reg = cv.warpPerspective(photo, h, (width, height))
        mask2 = np.zeros(background.shape, dtype=np.uint8)
        roi_corners2 = np.int32(positions2)
        channel_count2 = background.shape[2]
        ignore_mask_color2 = (255,)*channel_count2
        cv.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
        mask2 = cv.bitwise_not(mask2)
        masked_image2 = cv.bitwise_and(background, mask2)
        final = cv.bitwise_or(im1Reg, masked_image2)
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
