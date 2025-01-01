import os

# DETECTION_NETWORK = 'mobile0.25'
DETECTION_NETWORK = 'resnet50'

DETECTION_MODEL_PATH = os.path.abspath('../../models/Resnet50_Final.pth')
# DETECTION_MODEL_PATH = os.path.abspath('../../models/mobilenet0.25_Final.pth')
DETECTION_MODEL_TAR_PATH = os.path.abspath("../../models/mobilenetV1X0.25_pretrain.tar")

RECOGNITION_MODEL_PATH = os.path.abspath('../../models/backbone_glint360k_r100.pth')
RECOGNITION_NETWORK = 'r100'
# DEVICE = 'cuda:0'
DEVICE = 'cpu'


# vector database config
# QDRANT_HOST = '192.168.97.67'
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = 'face_collection'