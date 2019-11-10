from app import app, db, fs, queue
from app.forms import *
from werkzeug.utils import secure_filename
from werkzeug.wsgi import wrap_file
from flask import request, jsonify, render_template, abort, redirect, send_file, send_from_directory, Response
from bson.objectid import ObjectId
from mimetypes import guess_type
import datetime
from app.model import DroneAlgo
from app.construct import Construct
from PIL import Image
from math import sqrt, cos, sin, acos, tan
from zipfile import ZipFile
import json
import os
from random import randint
battery = 0
photo = 0
main = Construct()
drone = 0
location = 0


def get_dist(a, b):
    return sqrt((a["x"] - b["x"]) * (a["x"] - b["x"]) + (a["y"] - b["y"]) * (a["y"] - b["y"]))


def get_good_angle(a, b):
    A = {"x": b["x"] - a["x"], "y": b["y"] - a["y"]}
    B = {"x": 1, "y": 0}
    C = {"x": 0, "y": 0}
    tmp = (A["x"] * B["x"] + A["y"]*B["y"]) / get_dist(A, C)
    if tmp < -1:
        tmp = -1
    if tmp > 1:
        tmp = 1
    angle = acos(tmp)
    if A["y"] < 0:
        angle = -1 * angle
    return angle


def is_good(x, y):
    return x > l and y > l and x + l < SIZE and y + l < SIZE


def get_good(x, y):
    x = x * 4032 // 800 + 984
    y = y * 3024 // 600 + 1488
    return x, y


def solve_dataset(old_im, obj):

    UPLOAD_FOLDER_DATASET = '/tmp/kek/'
    ZIP_DATASET = '/tmp/photoset.zip'
    IMAGE_URL_DATASET = '/tmp/lol.jpg'
    SIZE = 6000
    if not os.path.exists(UPLOAD_FOLDER_DATASET):
        os.makedirs(UPLOAD_FOLDER_DATASET)
    angle_x = obj["angle_x"]
    angle_y = obj["angle_y"]
    h = obj["h"]
    points = obj["points"]
    overlapping = obj["overlapping"]
    lx = 2 * h * tan(angle_x / 360 * 3.1415926)
    ly = 2 * h * tan(angle_y / 360 * 3.1415926)
    res = []
    base = {"lat": 50.015781, "lng": 36.222751}  # Ivriy's house
    phi = base["lat"] / 180 * 3.1415926
    dpx = 111.321*cos(phi)
    dpy = 111.143
    top = UPLOAD_FOLDER_DATASET
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(top, name))
        for name in dirs:
            os.rmdir(os.path.join(top, name))

    a = 0
    try:
        zipfile.ZipFile(ZIP_DATASET, 'w')
    except:
        a = 1
    try:
        os.remove(ZIP_DATASET)
    except:
        a = 2
    old_size = old_im.size

    new_size = (6000, 6000)
    new_im = Image.new("RGB", new_size)
    new_im.paste(old_im, ((new_size[0]-old_size[0])//2,
                          (new_size[1]-old_size[1])//2))

    # new_im.save(IMAGE_URL_DATASET)
    cur = 0
    for point in points:
        point["x"], point["y"] = get_good(point["x"], point["y"])
    cur = points[0].copy()
    cur_i = 0
    nd = 0
    photo_id = 0
    images = []
    while cur_i + 1 < len(points):
        if get_dist(points[cur_i + 1], cur) < nd:
            nd = 0
            cur["x"] = points[cur_i+1]["x"]
            cur["y"] = points[cur_i+1]["y"]
            angle = get_good_angle(points[cur_i], points[cur_i + 1])

            rotate_angle = (90 + int(angle / 3.1415926 * 180) + 360) % 360
            phi = -rotate_angle / 180 * 3.1415926
            cur["x"] -= 3000
            cur["y"] -= 3000

            x = int(cur["x"] * cos(phi) - cur["y"] * sin(phi))
            y = int(cur["x"] * sin(phi) + cur["y"] * cos(phi))
            x += 3000
            y += 3000
            cur["x"] += 3000
            cur["y"] += 3000

            #img = Image.open(IMAGE_URL_DATASET)
            img = new_im.copy()
            tmp = str(photo_id)
            while len(tmp) < 4:
                tmp = '0' + tmp
            img.rotate(rotate_angle).crop((x - lx//2, y - lx//2, x + ly//2,
                                           y + ly//2)).save(UPLOAD_FOLDER_DATASET + 'result_' + tmp + '.jpg')
            photo_id += 1
            res.append({"alt_rel": h, "yaw": 90+angle / 3.1415926 * 180, "pitch": 0, "roll": 0,
                        "lng": cur["x"] / 1000 / dpx + base["lng"], "lat": base["lat"] - cur["y"] / 1000 / dpy})
            res[-1]["lat"] *= 1e7
            res[-1]["lng"] *= 1e7
            res[-1]["lat"] = int(res[-1]["lat"])
            res[-1]["lng"] = int(res[-1]["lng"])

            cur_i = cur_i + 1
            if cur_i + 1 == len(points):
                break

        else:
            angle = get_good_angle(points[cur_i], points[cur_i + 1])
            cur["x"] += nd * cos(angle)
            cur["y"] += nd * sin(angle)
            rotate_angle = (90 + int(angle / 3.1415926 * 180) + 360) % 360
            phi = -rotate_angle / 180 * 3.1415926

            cur["x"] -= 3000
            cur["y"] -= 3000
            x = int(cur["x"] * cos(phi) - cur["y"] * sin(phi))
            y = int(cur["x"] * sin(phi) + cur["y"] * cos(phi))
            x += 3000
            y += 3000
            cur["x"] += 3000
            cur["y"] += 3000

            #img = Image.open(IMAGE_URL_DATASET)
            img = new_im.copy()
            res.append({"alt_rel": h, "yaw": 90+angle / 3.1415926 * 180, "pitch": 0, "roll": 0,
                        "lng": cur["x"] / 1000 / dpx + base["lng"], "lat": base["lat"] - cur["y"] / 1000 / dpy})
            res[-1]["lat"] *= 1e7
            res[-1]["lng"] *= 1e7
            res[-1]["lat"] = int(res[-1]["lat"])
            res[-1]["lng"] = int(res[-1]["lng"])
            tmp = str(photo_id)
            while len(tmp) < 4:
                tmp = '0' + tmp
            img.rotate(rotate_angle).crop((x - lx//2, y - lx//2, x + ly//2,
                                           y + ly//2)).save(UPLOAD_FOLDER_DATASET + 'result_' + tmp + '.jpg')
            photo_id += 1
            nd = lx * (1 - overlapping)
    # images[0].show()
    for i in range(0, len(res)):
        new_pos = {"meta": {"type": "CAMERA_FEEDBACK"}, "data": res[i].copy()}
        res[i] = new_pos.copy()
    with open(UPLOAD_FOLDER_DATASET + 'logs.json', 'w') as outfile:
        for i in range(0, len(res)):
            print(json.dumps(res[i]), file=outfile)
    top = UPLOAD_FOLDER_DATASET

    with ZipFile(ZIP_DATASET, 'a') as zipObj:
        for folderName, subfolders, filenames in os.walk(top, topdown=False):
            for filename in filenames:
                filePath = os.path.join(folderName, filename)
                zipObj.write(filePath, '/' + filename)

    url = '/tmp/photoset.zip?'

    url = url + str(randint(0, 1000000))
    return url


@app.route('/send', methods=['GET', 'POST'])
def main():
    points = request.get_json()
    old_im = Image.open('./basic.jpg')

    return solve_dataset(old_im, points)


@app.route('/tmp/<path:filename>')
def uploaded_file(filename):
    print(filename)
    return send_from_directory('C:/Users/Maxim/Desktop/zahar1/tmp/',
                               filename, as_attachment=True)


@app.route('/sendHomePosition', methods=['POST'])
def makeHome():
    global location
    location = request.get_json()
    return jsonify(location)


@app.route('/sendDroneAlgo', methods=['POST'])
def sendDroneAlgo():
    global main
    main = DroneAlgo()
    main.get_dps(location)

    main.setDrone(drone)
    main.initialize()
    main.points = request.get_json()
    main.solve()
    return jsonify(main.result)


@app.route('/sendConstructive', methods=['POST'])
def sendConstructive():
    global main
    main = Construct()
    main.get_dps(location)

    main.setDrone(drone)
    main.initialize()
    main.points = request.get_json()
    #main.nn = len(main.points)
    main.solve()
    return jsonify(main.result)


@app.route('/sendInfo', methods=['POST'])
def sendInfo():
    global drone
    drone = request.get_json()
    return jsonify(drone)

# END DRON COVERAGE


@app.route('/')
def index():
    return render_template('index.html', title='Home', user={'username': 'Miguel'})


@app.route('/chooseTaskType')
def chooseTaskType():
    return render_template('chooseTaskType.html', title="Task Creation")


@app.route("/newTask/<taskType>", methods=['GET', 'POST'])
def newTask(taskType):
    if taskType == "task1":
        return render_template("task1.html")
    elif taskType == "task2":
        form = Task2Form()
    elif taskType == "task3":
        form = Task3Form()
    elif taskType == "task4":
        form = Task4Form()
    elif taskType == "task5":
        form = Task5Form()
    elif taskType == "task6":
        form = Task6Form()

    if form.validate_on_submit():
        if taskType == "task4":
            f = form.infile.data
            filename = secure_filename(f.filename)
            content_type, _ = guess_type(filename)
            f_id = fs.put(f, filename=filename, content_type=content_type)

            task = {
                "input_file": f_id,
                "created_at": datetime.datetime.utcnow(),
                "task_type": taskType,
                "state": "IDLE",
                "etc": {'alpha': form.alpha.data,
                        'beta': form.beta.data,
                        'zoom': form.zoom.data}
            }
            taskId = db.tasks.insert_one(task).inserted_id

            print("INFO: QUEUED TASK " + str(taskId))
            rq_task = queue.enqueue('app.tasks.'+taskType, taskId)
            task["redis_task"] = rq_task
        elif taskType == "task6":
            f = form.infile.data
            filename = secure_filename(f.filename)
            content_type, _ = guess_type(filename)
            f_id = fs.put(f, filename=filename, content_type=content_type)

            task = {
                "input_file": f_id,
                "created_at": datetime.datetime.utcnow(),
                "task_type": taskType,
                "state": "IDLE",
            }
            taskId = db.tasks.insert_one(task).inserted_id

            print("INFO: QUEUED TASK " + str(taskId))
            rq_task = queue.enqueue('app.tasks.'+taskType, taskId)
            task["redis_task"] = rq_task
        elif taskType == "task5":
            f = form.infile.data
            filename = secure_filename(f.filename)
            content_type, _ = guess_type(filename)
            f_id = fs.put(f, filename=filename, content_type=content_type)

            task = {
                "input_file": f_id,
                "created_at": datetime.datetime.utcnow(),
                "task_type": taskType,
                "state": "IDLE",
                "etc": {'isField': form.isField.data}
            }
            taskId = db.tasks.insert_one(task).inserted_id

            print("INFO: QUEUED TASK " + str(taskId))
            rq_task = queue.enqueue('app.tasks.'+taskType, taskId)
            task["redis_task"] = rq_task
        elif taskType == "task2":
            f = form.infile.data
            filename = secure_filename(f.filename)
            content_type, _ = guess_type(filename)
            f_id = fs.put(f, filename=filename, content_type=content_type)

            task = {
                "input_file": f_id,
                "created_at": datetime.datetime.utcnow(),
                "task_type": taskType,
                "state": "IDLE",
            }
            taskId = db.tasks.insert_one(task).inserted_id

            print("INFO: QUEUED TASK " + str(taskId))
            rq_task = queue.enqueue('app.tasks.'+taskType, taskId)
            task["redis_task"] = rq_task

        return redirect('/viewDetails/' + str(taskId))
    return render_template("newTask.html", title=taskType, taskType=taskType, form=form)


@app.route("/deleteTask/<taskId>")
def deleteTask(taskId):
    db.tasks.delete_one({"_id": ObjectId(taskId)})
    return redirect("/history")


@app.route("/viewDetails/<taskId>")
def viewDetails(taskId):
    task = db.tasks.find_one({"_id": ObjectId(taskId)})
    if task is None:
        return abort(404)
    return render_template('viewDetails.html', task=task)


@app.route("/getTaskInputFile/<taskId>")
def getTaskInputFile(taskId):
    task = db.tasks.find_one({"_id": ObjectId(taskId)})
    fileobj = fs.get(task["input_file"])
    data = wrap_file(request.environ, fileobj, buffer_size=1024 * 255)
    response = app.response_class(
        data,
        mimetype=fileobj.content_type,
        direct_passthrough=True,
    )
    response.content_length = fileobj.length
    response.last_modified = fileobj.upload_date
    response.set_etag(fileobj.md5)
    response.cache_control.max_age = 31536000
    response.cache_control.public = True
    response.make_conditional(request)
    return response


@app.route("/getTaskOutputFile/<taskId>")
def getTaskOutputFile(taskId):
    task = db.tasks.find_one({"_id": ObjectId(taskId)})
    fileobj = fs.get(task["output_file"])
    data = wrap_file(request.environ, fileobj, buffer_size=1024 * 255)
    response = app.response_class(
        data,
        mimetype=fileobj.content_type,
        direct_passthrough=True,
    )
    response.content_length = fileobj.length
    response.last_modified = fileobj.upload_date
    response.set_etag(fileobj.md5)
    response.cache_control.max_age = 31536000
    response.cache_control.public = True
    response.make_conditional(request)
    return response


@app.route('/history')
def history():
    tasks = db.tasks.find()
    return render_template('history.html', title="History", tasks=tasks)


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')
