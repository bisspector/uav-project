import numpy
from numpy import ndarray
import math
from collections import namedtuple
from math import cos, tan, sqrt, sin
from app.model import Geometry
from random import shuffle
Pair = namedtuple("Pair", ["first", "second"])
EPS = 1e-9

class Construct(Geometry):

    base, zero = {"lat": 0, "lng": 0}, {"x": 0, "y": 0}
    dpx, dpy = 1, 1
    lx, ly, max_h = 100, 100, 0
    points, a, path, ans_points, all_points = [], [], [], [], []
    cur_ans = 0
    drone, result = {}, {}

    def get_dps(self, location):
        self.base = location
        phi = self.base["lat"] / 180 * 3.1415926
        self.dpx = 111.321*cos(phi) - 0.0094*cos(3*phi)
        self.dpy = 111.143

    def get_distance(self):
        self.a.clear()
        for i in range(0, len(self.points)):
            self.a.append({"x": (self.points[i]["lng"] - self.base["lng"])*self.dpx*1000, "y": (self.points[i]["lat"] - self.base["lat"])*self.dpy*1000, "id": i})
        
        ###print(self.a)

    def setDrone(self, drone):
        self.drone = drone

    def initialize(self):
        self.drone = self.drone
        self.dd = (2 * self.drone["height"] * tan(self.drone["angle_y"] / 360 * 3.1415926))
        self.dd *= (1 - self.drone["overlapping"])
        self.ly = self.dd
        self.lx = (2 * self.drone["height"] * tan(self.drone["angle_x"] / 360 * 3.1415926))

    def make_ans(self):
        self.ans_points.clear()
        ###print(self.all_points)
        ###print(self.all_points)
        ###print(self.zero)
        
        for i in range(0, len(self.all_points)):
            self.ans_points.append({"lat": self.all_points[i]["y"] / 1000 / self.dpy + self.base["lat"], "lng": self.all_points[i]["x"] / 1000 / self.dpx + self.base["lng"]})
        #self.ans_points.insert(0, self.base)
        #self.ans_points.append(self.base)
        ###print(self.ans_points)
        

    def solve_good(self, points):
        a, n = self.Convex_hull(points.copy(), len(points))
        ###print(a)
        d = self.dd
        pt1, pt2 = self.findmax(a, n)
        aa, b, c = self.makeline(pt1, pt2)

        k = math.sqrt(aa * aa + b * b) * d;

        fl = 1
        j = 0
        ans = []
        ans.append(pt1.copy())
        #ans[0]["need"] = 0
        ans.append(pt2.copy())
        #ans[1]["need"] = 1
        sa = 2
        while (fl == 1):
            j = j - 1
            ass, sz1 = self.check(a, n, aa, b, c + j * k)
            if sz1 == 1:
                ans.append(ass[0])
                sa = sa + 1
                continue
            if sz1 == 0:
                break
            ax, ay = self.makevector(ans[sa - 1], ans[sa - 2])
            bx, by = self.makevector(ans[sa - 1], ass[0])
            cx, cy = self.makevector(ans[sa - 1], ass[1])
            cos1 = cs(ax, ay, bx, by);
            cos2 = cs(ax, ay, cx, cy);
            ###print("kukarek")
            if (cos1 > cos2):
                ans.append(ass[1].copy())
                ans.append(ass[0].copy())
            else:
                ans.append(ass[0].copy())
                ans.append(ass[1].copy())
            sa = sa + 2


        fl = 1
        j = 0
        while (fl == 1):
            j = j + 1
            ass, sz1 = self.check(a, n, aa, b, c + j * k)
            ###print(sz1)
            if sz1 == 1:
                ans.append(ass[0].copy())
                sa = sa + 1
                continue
            if sz1 == 0:
                break
            ax, ay = self.makevector(ans[sa - 1], ans[sa - 2])
            bx, by = self.makevector(ans[sa - 1], ass[0])
            cx, cy = self.makevector(ans[sa - 1], ass[1])
            ###print(ans[sa - 1]["x"], ans[sa - 1]["y"], sep = ' ')
            ###print(ass[0]["x"], ass[0]["y"], sep = ' ')
            ###print(ass[1]["x"], ass[1]["y"], sep = ' ')
            cos1 = self.cs(ax, ay, bx, by);
            cos2 = self.cs(ax, ay, cx, cy);
            t1 = cos1
            t2 = cos2
            if (cos1 > cos2):
                ans.append(ass[1].copy())
                ans.append(ass[0].copy())
            else:
                ans.append(ass[0].copy())
                ans.append(ass[1].copy())
            sa = sa + 2

        ans.append(ans[len(ans) - 1].copy())
        ###print("Zdarova")
        ###print(ans)
        #for i in range(sa):
        #    ##print(ans[i]["x"], ans[i]["y"], sep = ' ')
        ###print("U menya vse!")
        ans[0]["need"] = 0
        for i in range (0, len(ans)):
            ans[i]["need"] = 1
        #print("ololoshenk", ans)
        return ans


    def rec(self, a):
        n = len(a)
        tmp = self.big_area(a)
        if tmp < 0:
            b = []
            for i in range(-len(a) + 1, 1):
                b.append(a[-i])
            a = b.copy()
        if n == 3:
            b = self.solve_good(a.copy())
            #print("kek", b)
            return b.copy()
        ind = -1
        q = []
        for i in range(2, n):
            q.append(i)
        #shuffle(q)
        for i in q:
            if self.get_square(a[i - 2], a[i - 1], a[i]) < 0:
                ind  = i - 1
                break
        if self.get_square(a[n - 2], a[n - 1], a[0]) < 0:
            ind = n - 1
        if self.get_square(a[n - 1], a[0], a[1]) < 0:
            ind = 0
            ##print("kek", self.get_square(a[n - 1], a[0], a[1]))
         
        ##print(ind)
        ##print(a)
        if ind == -1:
            a = self.solve_good(a.copy())
            #print("medium kek", a)
            return a
        q.append(0)
        q.append(1)
        #shuffle(q)
        for j in q:
            if j == ind or j == ind - 1 or j == ind + 1:
                continue
            if ind == n - 1 and j == 0:
                continue
            if ind == 0 and j == n - 1:
                continue
            if self.check1(a, ind, j):
                ##print("AAAA", ind, j)
                ##print(j)
                
                b, c = self.make_ar(a.copy(), ind, j)
                ##print(1, b)
                ##print(2, c)
                b = self.rec(b.copy())
                c = self.rec(c.copy())
                
                ##print("double kek", b)
                ##print(c)
                if len(b) == 0:
                    a = c
                if len(c) == 0:
                    a = b
                if len(b) > 0 and len(c) > 0:
                    a = b + c
                #print("hard kek", a)
                return a
        #print("NO")
        return aa
    
    def make_path(self):
        #self.path = self.ans_points.
        b = []
        b.append(self.all_points[0].copy())
        for i in range (1, len(self.all_points)):
            if self.all_points[i]["x"] != self.all_points[i - 1]["x"]:
                b.append(self.all_points[i].copy())
        print("KEKKK", b)
        
        self.all_points = b.copy();
        cur = {"x": 0, "y": 0, "is": 0}
        cur_i = 0
        nd = 0
        lst = 0
        self.path.clear()
        ##print(self.all_points)
        self.path.append(cur.copy())
        while cur_i + 1 < len(self.all_points):
            print(cur_i, nd, cur, self.all_points[0], self.all_points[1])
            '''
            if cur_i == 1:
                cur["is"] = 1
            if cur_i + 1 == len(self.all_points) - 1:
                cur["is"] = 0
            if self.all_points[cur_i + 1]["need"] == 0:
                cur["is"] = 0
            if self.get_dist(self.all_points[cur_i + 1], cur) < nd:
                nd = nd - self.get_dist(self.all_points[cur_i + 1], cur)
                cur["x"] = self.all_points[cur_i+1]["x"]
                cur["y"] = self.all_points[cur_i+1]["y"]
                tmp = cur["is"]
                cur["is"] = 0
                self.path.append(cur.copy())
                cur["is"] = tmp
                cur_i = cur_i + 1
                if self.all_points[cur_i]["need"] == 0:
                    lst = 0
                    cur["is"] = 0
                else:
                    lst = lst + 1
                    if lst % 2 == 1:
                        cur["is"] = 1
                        nd = 0
                    else:
                        cur["is"] = 1
            '''
            if self.get_dist(self.all_points[cur_i + 1], cur) < nd:
                nd = 0
                cur["x"] = self.all_points[cur_i+1]["x"]
                cur["y"] = self.all_points[cur_i+1]["y"]
                cur["is"] = 0
                if nd >= self.ly / 2:
                    cur["is"] = 1
                self.path.append(cur.copy())
                cur_i = cur_i + 1
                if cur_i + 1 == len(self.all_points):
                    break
                if self.all_points[cur_i]["need"] == 0:
                    lst = 0
                    cur["is"] = 0
                else:
                    lst = lst + 1
                    if lst % 2 == 1:
                        cur["is"] = 1
                    else:
                        cur["is"] = 0
            else:
                ###print(self.all_points[cur_i], self.all_points[cur_i+1])
                angle = self.get_good_angle(self.all_points[cur_i], self.all_points[cur_i + 1])
                ###print(angle)
                cur["x"] += nd * cos(angle)
                cur["y"] += nd * sin(angle)
                nd = self.lx
                ###print("what")
                ###print(cur)
                self.path.append(cur.copy())
                ###print(self.path)
        b = []
        ###print(self.path)
        for i in range(0, len(self.path)):
            b.append({"lat": self.path[i]["y"] / 1000 / self.dpy + self.base["lat"], "lng": self.path[i]["x"] / 1000 / self.dpx + self.base["lng"], "is": self.path[i]["is"]})
        self.path = b.copy()
            
    def solve(self):
        
        self.all_points.clear()
        self.get_distance()
        ans = []
        cur = []
        a = self.a
        '''
        pmin = 0
        for i in range(1, len(a)):
            if a[i]["x"] < a[pmin]["x"]:
                pmin = i
        b = []
        for i in range(pmin, len(a)):
            b.append(a[i])
        for i in range(0, pmin):
            b.append(a[i])
        a = b.copy()'''
        ###print(a)
        ###print("begin")
        ans = self.rec(a)
        ###print(ans)
        ###print(len(ans))
        self.all_points = ans
        self.zero = {"x": 0, "y": 0, "need": 0}
        self.all_points.insert(0, self.zero)
        self.all_points.append(self.zero)
        ##print(self.all_points)
        self.make_path()
        ###print("kek")
        ###print(self.path)
        self.make_ans()
        ###print(len(self.all_points))
        ###print(len(self.ans_points))
        #self.path = self.ans_points
        self.result["way"] = self.path
        self.result["path"] = self.ans_points
        self.result["time"] = 0
        self.zero = {"x": 0, "y": 0}
        for i in range(1, len(self.all_points)):
            self.result["time"] += self.get_dist(self.all_points[i - 1], self.all_points[i]) / self.drone["speed"]
        self.result["time"] += (self.get_dist(self.zero, self.all_points[0]) + self.get_dist(self.all_points[-1], self.zero)) / self.drone["speed"]
        self.result["ok"] = 1
        if self.result["time"] / self.drone["speed"] > self.drone["battery"] / self.drone["perkm"]:
            self.result["ok"] = 0
        self.result["height"] = 1000
        self.result["needbattery"] = self.result["time"] / self.drone["speed"] * self.drone["perkm"] 