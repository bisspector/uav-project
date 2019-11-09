from app import app
import sys
from flask import jsonify, render_template, request, Response
from math import cos, sqrt, exp, tan, acos
import math
from random import random, randint, shuffle
import json
import numpy
from numpy import ndarray
from collections import namedtuple


Pair = namedtuple("Pair", ["first", "second"])
EPS = 1e-9

class Line(object):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class Geometry():
    #Distance between 2 points
    def get_dist(self, a, b):
        return sqrt((a["x"] - b["x"]) * (a["x"] - b["x"]) + (a["y"] - b["y"]) * (a["y"] - b["y"]))

    def sort(self, a, n):
        for i in range(n):
            for j in range(n - 1):
                if a[j]["x"] > a[j + 1]["x"]:
                    t = a[j]
                    a[j] = a[j + 1]
                    a[j + 1] = t
                if a[j]["x"] == a[j + 1]["x"] and a[j]["y"] > a[j + 1]["y"]:
                    t = a[j]
                    a[j] = a[j + 1]
                    a[j + 1] = t
        return a

    def cw(self, a, b, c):
        return a["x"] * (b["y"] - c["y"]) + b["x"] * (c["y"] - a["y"]) + c["x"] * (a["y"] - b["y"]) < 0

    def ccw(self, a, b, c):
        return a["x"] * (b["y"] - c["y"]) + b["x"] * (c["y"] - a["y"]) + c["x"] * (a["y"] - b["y"]) > 0


    def Convex_hull(self, a, n):
        a = self.sort(a, n)
        p1 = a[0]
        p2 = a[n - 1]
        up = []
        down = []
        su = 1
        sd = 1
        up.append(p1)
        down.append(p1)
        for i in range(1, n):
            if (i == n - 1 or self.cw(p1, a[i], p2)):
                while (su >= 2 and self.cw(up[su - 2], up[su - 1], a[i]) == 0):
                    up.pop()
                    su = su - 1
                up.append(a[i])
                su = su + 1
            if (i == n - 1 or self.ccw(p1, a[i], p2)):
                while (sd >= 2 and self.ccw(down[sd - 2], down[sd - 1], a[i]) == 0):
                    down.pop()
                    sd = sd - 1
                down.append(a[i])
                sd = sd + 1
        a.clear()
        sz = 0
        for i in range(su):
            a.append(up[i])
            sz = sz + 1
        for i in range(sd - 2, 0, -1):
            a.append(down[i])
            sz = sz + 1
        return a, sz

    def dist(self, a, b):
        return math.sqrt((a["x"] - b["x"]) * (a["x"] - b["x"]) + (a["y"] - b["y"]) * (a["y"] - b["y"]))

    def findmax(self, a, n):
        id = 0
        mx = 0
        for i in range(n - 1):
            d = self.dist(a[i], a[i + 1])
            if d > mx:
                mx = d
                id = i

        d = self.dist(a[n - 1], a[0])
        if d > mx:
            return a[n - 1], a[0]
        else:
            return a[id], a[id + 1]

    def makeline(self, p, q):
        a = p["y"] - q["y"]
        b = q["x"] - p["x"]
        c = -a * p["x"] - b * p["y"]
        return a, b, c

    def det(self, a, b, c, d):
        return a * d - b * c

    def parallel(self, m, n):
        return abs(self.det(m.a, m.b, n.a, n.b)) < EPS

    def intersect(self, m, n):
        zn = self.det(m.a, m.b, n.a, n.b)
        res = self.zero
        res["x"] = - self.det (m.c, m.b, n.c, n.b) / zn
        res["y"] = - self.det (m.a, m.c, n.a, n.c) / zn
        return res

    def check(self, ar, n, a, b, c):
        l = Line(a, b, c)
        ar.append(ar[0])
        n = n + 1
        ans = []
        sz = 0
        q = set()
        for i in range(n - 1):
            a1, b1, c1 = self.makeline(ar[i], ar[i + 1])
            l1 = Line(a1, b1, c1)
            if (self.parallel(l, l1) == 0):
                pt = self.intersect(l, l1)
                if pt["x"] >= min(ar[i]["x"], ar[i + 1]["x"]) and pt["x"] <= max(ar[i]["x"], ar[i + 1]["x"]):
                    if pt["y"] >= min(ar[i]["y"], ar[i + 1]["y"]) and pt["y"] <= max(ar[i]["y"], ar[i + 1]["y"]):
                        q.add(Pair(pt["x"], pt["y"]))

        for i in q:
            t = i
            tt = {"x":t.first, "y":t.second}
            ans.append(tt)
            sz = sz + 1
        return ans, sz

    def orarea(self, x1, y1, x2, y2, x3, y3):
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    def big_area(self, ar):
        #print("DO")
        #print(ar)
        ox = 1e7
        oy = 1e7
        n = len(ar)
        s = 0
        for i in range(n - 1):
            s = s + self.orarea(ox, oy, ar[i]["x"], ar[i]["y"], ar[i + 1]["x"], ar[i + 1]["y"])
        s = s + self.orarea(ox, oy, ar[n - 1]["x"], ar[n - 1]["y"], ar[0]["x"], ar[0]["y"])
        #print("-------------")
        #print(ar)
        #print("AFTER")
        return s

    def make_ar(self, ar, id1, id2):
        n = len(ar)
        #print("OOOuuu")
        #print(ar)
        a = []
        b = []
        if id1 > id2:
            c = id1
            id1 = id2
            id2 = c
        #print(id1, id2, sep = ' ')
        for j in range(id2, n):
            b.append(ar[j])
        for j in range(0, id1 + 1):
            b.append(ar[j])
        for j in range(id1, id2 + 1):
            a.append(ar[j])
        #print("kuku")
        #print(a)
        #print(b)
        #print("kukarek")
        return a, b
    
    def intersection(self, a, b, c, d):
        if self.get_square(a, c, b) * self.get_square(a, d, b) < 0 and self.get_square(c, a, d) * self.get_square(c, b, d) < 0:
            return 1
        return 0
    def check1(self, arr, i1, i2):
        n = len(arr)
        '''
        if id1 > id2:
            c = id1
            id1 = id2
            id2 = c
        a = []
        b = []
        for j in range(id2, n):
            b.append(arr[j])
        for j in range(0, id1 + 1):
            b.append(arr[j])
        for j in range(id1, id2 + 1):
            a.append(arr[j])
        sful = abs(self.bigarea(arr))
        s1 = abs(self.bigarea(a))
        s2 = abs(self.bigarea(b))
        print("kek", s1, s2, sful)
        print(a)
        print(b)
        print(arr)
        if abs(s1 + s2 - sful) < EPS:
            return 1
        else:
            return 0
        '''
        kol = 0
        for i in range (0, len(arr) - 1):
            if self.intersection(arr[i1], arr[i2], arr[i], arr[i + 1]) == 1:
                kol = kol + 1
        if self.intersection(arr[0], arr[-1], arr[i1], arr[i2]) == 1:
            kol = kol + 1
        if kol > 0:
            return 0
        '''
        x1 = arr[i1 - 1]["x"] - arr[i1]["x"]
        x2 = arr[i2]["x"] - arr[i1]["x"]
        x3 = arr[(i1 + 1) % n]["x"] - arr[i1]["x"]
        if x3 > x1:
            x1, x3 = x3, x1
        if x1 <= x2 and x2 <= x3:
            return 0
        else:
            return 1
        '''
        if self.big_area(arr) > 0:
            a1, a2, a3 = self.get_good_angle(arr[i1], arr[i1 - 1]), self.get_good_angle(arr[i1], arr[i2]), self.get_good_angle(arr[i1], arr[(i1 + 1) % len(arr)])
            if a1 < 0:
                a1 += 2 * 3.1415926
            
            if a2 < 0:
                a2 += 2 * 3.1415926
                
            if a3 < 0:
                a3 += 2 * 3.1415926
            return a2 < a1 or a2 > a3
        else:
            a1, a2, a3 = self.get_good_angle(arr[i1], arr[i1 - 1]), self.get_good_angle(arr[i1], arr[i2]), self.get_good_angle(arr[i1], arr[(i1 + 1) % len(arr)])
            if a1 < 0:
                a1 += 2 * 3.1415926
            
            if a2 < 0:
                a2 += 2 * 3.1415926
                
            if a3 < 0:
                a3 += 2 * 3.1415926
            return a2 < a3 or a2 > a1
    
    def cs(self, ax, ay, bx, by):
        return (ax * bx + ay * by) / (math.sqrt(ax * ax + ay * ay) * math.sqrt(bx * bx + by * by))

    def makevector(self, p, q):
        x = q["x"] - p["x"]
        y = q["y"] - p["y"]
        return x, y

    #angle
    def get_angle(self, a, b, c):
        A = {"x": b["x"] - a["x"], "y": b["y"] - a["y"]}
        B = {"x": c["x"] - b["x"], "y": c["y"] - b["y"]}
        C = {"x": 0, "y": 0}
        #print(A)
        tmp = (A["x"] * B["x"] + A["y"]*B["y"]) / self.get_dist(A, C) / self.get_dist(B, C)
        if tmp < -1:
            tmp = -1
        if tmp > 1:
            tmp = 1
        angle = acos(tmp)
        if angle < 0:
            angle = angle + 2 * 3.1415926
        angle = angle / 3.1415926 * 360
        if 360 - angle < angle:
            angle = 360 - angle
        return angle
    
    def get_good_angle(self, a, b):
        A = {"x": b["x"] - a["x"], "y": b["y"] - a["y"]}
        B = {"x": 1, "y": 0}
        C = {"x": 0, "y": 0}
        
        #print(A["x"] * B["x"] + A["y"]*B["y"], self.get_dist(A, C))
        tmp = (A["x"] * B["x"] + A["y"]*B["y"]) / self.get_dist(A, C)
        if tmp < -1:
            tmp = -1
        if tmp > 1:
            tmp = 1
        #print(tmp)
        angle = acos(tmp)
        if A["y"] < 0:
            angle = -1 * angle
        return angle
    
    
    #Oriented square of triangle
    
    
    def get_square(self, a, b, c):
        return (a["x"] - b["x"]) * (a["y"] + b["y"]) + (b["x"] - c["x"]) * (b["y"] + c["y"]) + (c["x"] - a["x"]) * (c["y"] + a["y"])

    def is_good_polygon(self, a):
        ссс = 0
        if self.get_square(a[0], a[1], a[2]) < 0:
            ccc = -1
        else:
            ccc = 1
        for i in range(1, len(a) - 2):
            qqq = 0
            if self.get_square(a[i], a[i + 1], a[i + 2]) < 0:
                qqq = -1
            else:
                qqq = 1
            if ccc != qqq:
                return 0
        qqq = 0
        if self.get_square(a[-1], a[0], a[1]) < 0:
            qqq = -1
        else:
            qqq = 1
        if ccc != qqq:
            return 0
        qqq = 0
        if self.get_square(a[-2], a[-1], a[0]) < 0:
            qqq = -1
        else:
            qqq = 1
        if ccc != qqq:
            return 0
        return 1


class DroneAlgo(Geometry):
    base, zero = {"lat": 0, "lng": 0}, {"x": 0, "y": 0}
    dpx, dpy = 1, 1
    lx, ly, max_h = 100, 100, 0
    points, a, ans_points, all_points = [], [], [], []
    cur_ans = 0
    drone, result = {}, {}

    #GEOMETRY
    #Get dpx and dpy
    def get_dps(self, location):
        self.base = location
        phi = self.base["lat"] / 180 * 3.1415926
        self.dpx = 111.321*cos(phi) - 0.0094*cos(3*phi)
        self.dpy = 111.143

    #Transformation from lat/lng to cartesian coordinates
    def get_distance(self):
        self.a.clear()
        for i in range(0, len(self.points)):
            self.a.append({"x": (self.points[i]["lng"] - self.base["lng"])*self.dpx*1000, "y": (self.points[i]["lat"] - self.base["lat"])*self.dpy*1000, "id": i})


    #Check is the point inside the border
    def is_in(self, x, y):
        p = {"x": x, "y": y}
        q = {"x": 1000000000, "y": 970005041}
        kol = 0
        for i in range(0, len(self.a) - 1):
            if self.intersection(p, q, self.a[i], self.a[i + 1]) == 1:
                kol = kol + 1
        if self.intersection(p, q, self.a[0], self.a[-1]) == 1:
            kol = kol + 1
        if kol % 2 == 1:
            return 1
        return 0

    #Check do the segment and the border intersect
    def inter(self, x1, y1, x2, y2):
        p = {"x": x1, "y": y1}
        q = {"x": x2, "y": y2}
        for i in range(0, len(self.a) - 1):
            if self.intersection(p, q, self.a[i], self.a[i + 1]) == 1:
                return 1
        if self.intersection(p, q, self.a[0], self.a[-1]) == 1:
            return 1
        return 0

    #Check do we need the square lx * ly with left-down in (x, y)
    def good_square(self, x, y):
        if self.is_in(x, y) == 1 or self.is_in(x + self.lx, y) == 1 or self.is_in(x + self.lx, y + self.ly) == 1 or self.is_in(x, y + self.ly) == 1:
            return 1
        if self.inter(x, y, x + self.lx, y) == 1 or self.inter(x + self.lx, y, x + self.lx, y + self.ly) == 1 or self.inter(x + self.lx, y + self.ly, x, y + self.ly) == 1 or self.inter(x, y + self.ly, x, y) == 1:
            return 1
        return 0


    #END OF GEOMETRY


    #TSP SOLUTION

    #Function of the state
    def f_optimal(self, all_points):
        dist, turn = 0, 0
        for i in range(0, len(all_points) - 1):
            dist = dist + self.get_dist(all_points[i], all_points[i + 1])
        #dist = dist + get_dist(all_points[0], all_points[-1])
        dist = dist + self.get_dist(self.zero, all_points[0]) + self.get_dist(all_points[-1], self.zero)
        '''
        for i in range(1, len(all_points) - 1):
            turn = turn + self.get_angle(all_points[i - 1], all_points[i], all_points[i + 1])
        '''
        return dist / self.drone["speed"]

    #Solving TSP
    def solve_TSP(self):
        shuffle(self.all_points)
        self.cur_ans = self.f_optimal(self.all_points)
        t1 = 200
        t2 = 0.00001
        while t1 > t2:
            p1 = randint(0, len(self.all_points) - 1)
            p2 = randint(0, len(self.all_points) - 1)
            if p1 > p2:
                p1, p2 = p2, p1

            b = []
            for i in range(0, p1):
                b.append(self.all_points[i])
            for i in range(p1, p2 + 1):
                b.append(self.all_points[p2 - i + p1])
            for i in range(p2 + 1, len(self.all_points)):
                b.append(self.all_points[i])
            newVal = self.cur_ans
            self.zero = {"x": 0, "y": 0}
            if p1 == 0:
                if p2 == len(self.all_points) - 1:
                    newVal = self.cur_ans
                else:
                    newVal = self.cur_ans - (self.get_dist(self.all_points[p2], self.all_points[p2 + 1])  - self.get_dist(self.all_points[p1], self.all_points[p2 + 1]) + self.get_dist(self.zero, self.all_points[p1]) - self.get_dist(self.zero, self.all_points[p2]))/ self.drone["speed"]
            else:
                if p2 == len(self.all_points) - 1:
                    newVal = self.cur_ans - (self.get_dist(self.all_points[p1 - 1], self.all_points[p1]) - self.get_dist(self.all_points[p2], self.all_points[p1 - 1]) + self.get_dist(self.zero, self.all_points[p2]) - self.get_dist(self.zero, self.all_points[p1]))/ self.drone["speed"]
                else:
                    newVal = self.cur_ans - (self.get_dist(self.all_points[p1 - 1], self.all_points[p1]) - self.get_dist(self.all_points[p2], self.all_points[p1 - 1]) + self.get_dist(self.all_points[p2], self.all_points[p2 + 1]) - self.get_dist(self.all_points[p1], self.all_points[p2 + 1]))/ self.drone["speed"]
            
            '''
            if p1 != p2 and p1 > 2:
                newVal = newVal - self.get_angle(self.all_points[p1 - 2], self.all_points[p1 - 1], self.all_points[p1]) / self.drone["speed"]+ self.get_angle(self.all_points[p1 - 2], self.all_points[p1 - 1], self.all_points[p2]) / self.drone["speed"]
            if p2 - 1 > p1 and p1 > 0:
                newVal = newVal - self.get_angle(self.all_points[p1 - 1], self.all_points[p1], self.all_points[p1+1]) / self.drone["speed"]+ self.get_angle(self.all_points[p1 - 1], self.all_points[p2], self.all_points[p2-1]) / self.drone["speed"]
            if p1 != p2 and p2 + 2 < len(self.all_points):
                newVal = newVal - self.get_angle(self.all_points[p2], self.all_points[p2+1], self.all_points[p2+2]) / self.drone["speed"]+ self.get_angle(self.all_points[p1], self.all_points[p2+1], self.all_points[p2+2]) / self.drone["speed"]
            if p1 + 1 <= p2 and p2 + 1 < len(self.all_points):
                newVal = newVal - self.get_angle(self.all_points[p2-1], self.all_points[p2], self.all_points[p2+1]) / self.drone["speed"]+ self.get_angle(self.all_points[p1+1], self.all_points[p1], self.all_points[p2+1]) / self.drone["speed"]
            '''


            opt = self.cur_ans - newVal
            if opt > 0 or random() < exp(min(opt / t1, 1)):
                self.cur_ans = newVal
                b = []
                for i in range(0, p1):
                    b.append(self.all_points[i])
                for i in range(p1, p2 + 1):
                    b.append(self.all_points[p2 - i + p1])
                for i in range(p2 + 1, len(self.all_points)):
                    b.append(self.all_points[i])
                self.all_points = b.copy()
            t1 *= 0.9998

    #END OF TSP SOLUTION

    def make_ans(self):
        self.ans_points = []
        self.all_points.insert(0, self.zero)
        self.all_points.append(self.zero)
        #print(len(self.all_points))
        for i in range(0, len(self.all_points)):
            self.ans_points.append({"lat": self.all_points[i]["y"] / 1000 / self.dpy + self.base["lat"], "lng": self.all_points[i]["x"] / 1000 / self.dpx + self.base["lng"]})
            #print(len(self.ans_points))

    def get_time(self):
        return self.cur_ans

    def solve(self):
        self.get_distance()
        self.all_points.clear()
        minx = 0
        miny = 0
        maxx = 0
        maxy = 0
        #print(self.drone)
        for point in self.a:
            minx = min(minx, point["x"])
            miny = min(miny, point["y"])
            maxx = max(maxx, point["x"])
            maxy = max(maxy, point["y"])
        k_up = int(max(0, maxy // self.ly + 1))
        k_down = int(max(0, -miny // self.ly + 1))
        k_left = int(max(0, -minx // self.lx + 1))
        k_right = int(max(0, maxx // self.lx + 1))
        if (k_up + k_down)*(k_left + k_right) > 50 * 50:
            self.result["ok"] = 2
            return
        for i in range(-k_left - 1, k_right + 1):
            for j in range(-k_down - 1, k_up + 1):
                if self.good_square(i * self.lx, j * self.ly) == 1:
                    self.all_points.append({"x": i * self.lx + self.lx / 2, "y" : j * self.ly + self.ly / 2})
        self.solve_TSP()
        self.make_ans()
        self.result["path"] = self.ans_points
        self.result["time"] = self.get_time()
        self.result["ok"] = 1
        if self.result["time"] / self.drone["speed"] / 1000 > self.drone["battery"] / self.drone["perkm"]:
            self.result["ok"] = 0
        self.result["needbattery"] = self.result["time"] / self.drone["speed"] * self.drone["perkm"] 
        self.result["way"] = self.ans_points;
        for i in range(1, len(self.result["way"])-1):
            self.result["way"][i]["is"] = 1
    
    def setDrone(self, d):
        self.drone = d

    def initialize(self):
        self.lx = (2 * self.drone["height"] * tan(self.drone["angle_x"] / 360 * 3.1415926))
        self.ly = (2 * self.drone["height"] * tan(self.drone["angle_y"] / 360 * 3.1415926))
        self.lx *= (1 - self.drone["overlapping"])
        self.ly *= (1 - self.drone["overlapping"])