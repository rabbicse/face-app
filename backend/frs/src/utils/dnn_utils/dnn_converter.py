import base64
import pickle


def encode_np(data):
    np_bytes = pickle.dumps(data)
    print(np_bytes)
    np_b64 = base64.encodebytes(np_bytes)
    print(np_b64)
    return np_b64.decode('utf-8')


def encode_np_hex(data):
    np_bytes = pickle.dumps(data)
    return np_bytes.hex()


def decode_np(b64_data):
    b64_bytes = b64_data.encode('ascii')
    data_bytes = base64.decodebytes(b64_bytes)

    return pickle.loads(data_bytes)


def decode_np_hex(hex_data):
    data_bytes = bytes.fromhex(hex_data)
    return pickle.loads(data_bytes)