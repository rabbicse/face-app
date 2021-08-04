import json
import logging
import os
import warnings

import cv2.cv2 as cv2
import numpy
from flask import Flask, request, jsonify
from flask_cors import CORS

from face_handler import FaceHandler
from utils.redis_handler import RedisHandler

warnings.filterwarnings("ignore")

app = Flask(__name__, static_url_path="/static")
CORS(app)

logger = logging.getLogger(__name__)

face_handler = FaceHandler()

redis_handler = RedisHandler()


@app.route('/detect/v1', methods=['POST'])
def detect_v1():
    photo = request.get_data()
    try:
        data = numpy.frombuffer(photo, dtype=numpy.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        faces = face_handler.detect_faces(frame)

        return jsonify({'status': 'success',
                        'faces': json.dumps(faces)})
    except Exception as x:
        logger.error(f'Error when detect faces by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


@app.route('/embedding/v1', methods=['POST'])
def extract_embedding_v1():
    photo = request.get_data()
    try:
        data = numpy.frombuffer(photo, dtype=numpy.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)
        return jsonify({'status': 0, 'embedding': emb.tolist()})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


@app.route('/enroll/v1', methods=['POST'])
def enroll_v1():
    photo = request.files.get('image')
    data = request.form.get('data')
    try:
        photo_data = photo.read()

        frame = numpy.frombuffer(photo_data, dtype=numpy.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)
        # print(emb)

        person_info = json.loads(data)
        name = person_info['name']
        print(name)

        # todo: insert database and get uniqe id
        redis_handler.insert_data(name, emb)
        return jsonify({'status': 0, 'embedding': emb.tolist()})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


@app.route('/match/v1', methods=['POST'])
def match_v1():
    photo = request.files.get('image')
    data = request.form.get('data')
    try:
        photo_data = photo.read()
        frame = numpy.frombuffer(photo_data, dtype=numpy.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)

        person_info = json.loads(data)
        name = person_info['name']

        # todo: insert database and get uniqe id
        dec_emb = redis_handler.search_data(name)
        print(dec_emb)

        # todo: match
        score = face_handler.match(emb, dec_emb)

        return jsonify({'status': 0, 'score': str(score)})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


@app.route('/test/v1', methods=['POST'])
def test_v1():
    data = request.form.get('data')
    try:
        return jsonify({'status': 0, 'score': str(data)})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


@app.route('/test/v2', methods=['GET'])
def test_v2():
    try:
        return jsonify({'status': 0, 'score': 0})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


def main():
    app.run(host='0.0.0.0', port=os.getenv('FR_PORT', 5000), debug=True)


if __name__ == "__main__":
    main()
