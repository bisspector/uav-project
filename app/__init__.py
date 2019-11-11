from flask import Flask
from flask.json import JSONEncoder
from config import Config
from flask_migrate import Migrate
from flask_cors import CORS
from pymongo import MongoClient
from gridfs import GridFS
from redis import Redis
import rq


app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r'/*': {'origins': '*'}})
mongo = MongoClient('mongodb://localhost:27017/')
db = mongo.uav
fs = GridFS(db)
queue = rq.Queue(connection=Redis.from_url('redis://'), default_timeout=3600)

from app import routes
