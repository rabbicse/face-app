from abc import abstractmethod

from utils.vision_utils.singleton_decorator import SingletonDecorator


# @SingletonDecorator
class CacheService:
    # def __init__(self,
    #              host: str,
    #              port: str):
    #     self.host = host
    #     self.port = port

    @abstractmethod
    def insert_data(self,
                    key: str,
                    value: object):
        pass

    @abstractmethod
    def insert_text(self,
                    key: str,
                    value: str):
        pass

    def search(self,
               key: str):
        pass

    @abstractmethod
    def get_all(self):
        pass
