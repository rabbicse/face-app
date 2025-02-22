import base64
import json
import os
import warnings
import cv2.cv2 as cv2
import numpy
from flask import Flask, request, jsonify
from flask_cors import CORS
from dnn_utils import dnn_converter
from api.services.face_service import FaceService
from vision_utils import log_utils
from vision_utils.redis_handler import RedisHandler

warnings.filterwarnings("ignore")
app = Flask(__name__, static_url_path="/static")
CORS(app)

logger = log_utils.LogUtils().get_logger(__name__)

face_handler = FaceService()

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
    """
    @return:
    """
    photo = request.files.get('photo')
    try:
        photo_data = photo.read()

        data = numpy.frombuffer(photo_data, dtype=numpy.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)
        if type(emb) is int and emb == -2:
            return jsonify({'status': 2, 'embedding': str(-3)})

        emb_bytes = emb.tobytes()
        emb_b64 = base64.b64encode(emb_bytes)
        embedding = emb_b64.decode('ascii')
        return jsonify({'status': 0, 'embedding': embedding})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


@app.route('/embedding/v2', methods=['POST'])
def extract_embedding_v2():
    photo = request.files.get('photo')
    try:
        photo_data = photo.read()

        data = numpy.frombuffer(photo_data, dtype=numpy.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)
        if type(emb) is int and emb == -2:
            return jsonify({'status': 2, 'embedding': str(-3)})

        # emb_bytes = emb.tobytes()
        # emb_b64 = base64.b64encode(emb_bytes)
        # embedding = emb_b64.decode('ascii')
        embedding = dnn_converter.encode_np_hex(emb)
        return jsonify({'status': 0, 'embedding': embedding})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': f'Unexpected Exception: {x}'})


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

        if type(emb) is int and emb == -2:
            return jsonify({'status': 2, 'score': str(-3)})

        person_info = json.loads(data)
        name = person_info['name']

        # todo: insert database and get uniqe id
        dec_emb = redis_handler.search_data(name)
        # print(dec_emb)

        # todo: match
        score = face_handler.match(emb, dec_emb)

        return jsonify({'status': 0, 'score': str(score)})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


@app.route('/match/v2', methods=['POST'])
def match_v2():
    photo = request.files.get('photo')
    embeddings = request.form.get('embeddings')
    try:
        photo_data = photo.read()
        frame = numpy.frombuffer(photo_data, dtype=numpy.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)

        if type(emb) is int and emb == -2:
            return jsonify({'status': 2, 'score': str(-3)})

        embeddings_data = json.loads(embeddings)
        embedding_b64 = embeddings_data['embedding']
        embedding = dnn_converter.decode_np_hex(embedding_b64)
        # print(embedding)
        # print('done....')

        # match
        score = face_handler.match(emb, embedding)

        # print(score)

        return jsonify({'status': 0, 'score': str(score)})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        # return abort()


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
