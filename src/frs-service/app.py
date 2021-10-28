import base64
import json
import os
import warnings
import cv2.cv2 as cv2
import numpy
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from dnn_utils import dnn_converter
from face_handler import FaceHandler
from vision_utils import log_utils
from vision_utils.decorators import TimeitDecorator
from vision_utils.redis_handler import RedisHandler

warnings.filterwarnings("ignore")

app = Flask(__name__, static_url_path="/static")
CORS(app)

logger = log_utils.LogUtils().get_logger(__name__)

DETECTOR_MODEL_PATH = os.path.abspath('models/mobilenet0.25_Final.pth')
DETECTOR_MODEL_TAR_PATH = os.path.abspath('models/mobilenetV1X0.25_pretrain.tar')
RECOGNIZER_MODEL_PATH = os.path.abspath('models/backbone-r100m.pth')

dnn_config = {
    'detector_model_path': DETECTOR_MODEL_PATH,
    'detector_model_tar_path': DETECTOR_MODEL_TAR_PATH,
    'recognizer_model_path': RECOGNIZER_MODEL_PATH,
    'recognizer_model_architecture': 'r100'
}
face_handler = FaceHandler(detector_network='mobile0.25',
                           dnn_config=dnn_config,
                           debug=False)

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
@TimeitDecorator()
def extract_embedding_v2():
    photo = request.files.get('photo')
    try:
        photo_data = photo.read()

        data = numpy.frombuffer(photo_data, dtype=numpy.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)
        if type(emb) is int and emb == -2:
            return Response('Face orientation is not perfect!', status=400)

        embedding = dnn_converter.encode_np_hex(emb)

        return jsonify({'status': 0, 'embedding': embedding})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        # return jsonify({'msg': f'Unexpected Exception: {x}'})
        return Response('Face orientation is not perfect!', status=500)


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

        # todo: insert database and get uniqe id
        redis_handler.insert_data(name, emb)
        return jsonify({'status': 0, 'embedding': emb.tolist()})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


@app.route('/match/v3', methods=['POST'])
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

        # todo: match
        score = face_handler.match(emb, dec_emb)
        print(f'Score: {score}')

        return jsonify({'status': 0, 'score': str(score)})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return jsonify({'msg': 'Not a valid image!'})


# @app.route('/match/v2', methods=['POST'])
# @TimeitDecorator()
# def match_v2():
#     photo = request.files.get('photo')
#     embeddings = request.form.get('embeddings')
#     try:
#         photo_data = photo.read()
#         frame = numpy.frombuffer(photo_data, dtype=numpy.uint8)
#         frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
#         emb = face_handler.extract_embedding(frame)
#
#         if type(emb) is int and emb == -2:
#             return Response('Face orientation is not perfect!', status=400)
#
#         embeddings_data = json.loads(embeddings)
#         embedding_b64 = embeddings_data['embedding']
#         embedding = dnn_converter.decode_np_hex(embedding_b64)
#
#         # match
#         score = face_handler.match(emb, embedding)
#
#         return jsonify({'status': 0, 'score': str(score)})
#     except Exception as x:
#         logger.error(f'Error when recognize by image. Details: {x}')
#         return Response('Face orientation is not perfect!', status=500)

@app.route('/match/v1', methods=['POST'])
@TimeitDecorator()
def match_v3():
    photo = request.files.get('image')
    data = request.form.get('data')
    try:
        photo_data = photo.read()
        frame = numpy.frombuffer(photo_data, dtype=numpy.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
        result = face_handler.extract_embeddings(frame)
        print(result)

        if not result:
            return Response('FRS engine error!', status=500)

        if result['status'] == 1:
            return Response('No face detected!', status=400)
        elif result['status'] == 2:
            return Response('Face orientation issue!', status=400)

        person_info = json.loads(data)
        name = person_info['name']

        # todo: insert database and get uniqe id
        dec_emb = redis_handler.search_data(name)

        # todo: match
        score = face_handler.match(result['results'][0], dec_emb)
        print(f'Score: {score}')

        return jsonify({'status': 0, 'score': str(score)})
    except Exception as x:
        logger.error(f'Error when recognize by image. Details: {x}')
        return Response('Face orientation is not perfect!', status=500)


def main():
    app.run(host='0.0.0.0', port=os.getenv('FR_PORT', 5000), debug=True)


if __name__ == "__main__":
    main()
