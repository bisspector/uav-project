from app import app, db, fs, queue
from app.forms import *
from werkzeug.utils import secure_filename
from werkzeug.wsgi import wrap_file
from flask import request, jsonify, render_template, abort, redirect
from bson.objectid import ObjectId
from mimetypes import guess_type
import datetime


@app.route('/')
def index():
    return render_template('index.html', title='Home', user={'username': 'Miguel'})


@app.route('/chooseTaskType')
def chooseTaskType():
    return render_template('chooseTaskType.html', title="Task Creation")


@app.route("/newTask/<taskType>", methods=['GET', 'POST'])
def newTask(taskType):
    if taskType == "task1":
        form = Task1Form()
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
