import glob
import logging
import os.path

import cv2

from recognition.arc_face import arc_face
from utils.vision_utils.decorators import timeit
from vector_db.qdrant_context import VectorDbContext

# from vectoe_db.qdrant_context import VectorDbContext

logger = logging.getLogger(__name__)

rec = arc_face.ArcFace(os.path.abspath('../models/backbone-r100m.pth'), model_architecture="r100")
vector_db = VectorDbContext()


def load_embeddings(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    frm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frm = cv2.resize(frm, (112, 112))
    emb = rec.get_embedding(frm)
    # print(emb)
    return emb


def find_images_one_by_one(base_path):
    # Use glob to find all .jpg files recursively
    for file_path in glob.glob(f"{base_path}/**/*.jpg", recursive=True):
        # Get the subfolder name relative to base_path
        subfolder = os.path.relpath(os.path.dirname(file_path), base_path).replace('_', ' ')
        yield subfolder, file_path  # Yield one result at a time


emb = load_embeddings(img_path='../samples/480.jpg')
vector_db.upsert(vector=emb, name='Mehedi', id=1)
# vector_db.search_embedding(vector=emb)


# id = 1
# base_path = 'D:/datasets/archive/lfw-funneled/lfw_funneled'
# for name, img_path in find_images_one_by_one(base_path=base_path):
#     print(f'{name} => {img_path}')
#     emb = load_embeddings(img_path=img_path)
#     vector_db.upsert(vector=emb, name=name, id=id)
#     id += 1
