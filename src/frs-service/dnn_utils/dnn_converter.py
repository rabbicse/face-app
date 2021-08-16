import base64
import pickle


def encode_np(data):
    np_bytes = pickle.dumps(data)
    np_b64 = base64.encodebytes(np_bytes)
    return np_b64.decode('ascii')


def decode_np(b64_data):
    b64_bytes = b64_data.encode('ascii')
    data_bytes = base64.decodebytes(b64_bytes)

    return pickle.loads(data_bytes)
