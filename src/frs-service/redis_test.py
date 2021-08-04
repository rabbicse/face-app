from utils.redis_handler import RedisHandler

redis_handler = RedisHandler()
redis_handler.insert_text('key', 'key')