import struct

import numpy as np
import redis

from vision_utils.singleton_decorator import SingletonDecorator


# https://stackoverflow.com/questions/55311399/fastest-way-to-store-a-numpy-array-in-redis

@SingletonDecorator
class RedisHandler:
    def __init__(self, host: str = '127.0.0.1', port: str = '6379'):
        self.client = redis.Redis(
            host=host,
            port=port
        )

    def insert_data(self, key, value):
        try:
            print(key)
            print(str(value.dtype))
            v = value.ravel().tostring()
            response = self.client.set(key, v)
            print(response)
        except Exception as x:
            print(x)

    def insert_text(self, key, value):
        try:
            response = self.client.set(key, value)
            print(response)
        except Exception as x:
            print(x)

    def search_data(self, key):
        try:
            n = self.client.get(key)
            if n:
                decoded = np.fromstring(n, dtype='float32')
                return decoded
        except Exception as x:
            print(x)

    def toRedis(self, n):
        """Store given Numpy array 'a' in Redis under key 'n'"""
        # print(n.shape)
        h = n.shape
        shape = struct.pack('>II', h[0], 0)
        encoded = shape + n.tobytes()
        return encoded

    def fromRedis(self, encoded):
        h, w = struct.unpack('>II', encoded[:8])
        # Add slicing here, or else the array would differ from the original
        n = np.frombuffer(encoded[8:]).reshape(h, w)
        return n
