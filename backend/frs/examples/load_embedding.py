import glob
import logging
import os.path
import uuid

import cv2

from api.model.person import Person
from recognition.arc_face import arc_face
from utils.vision_utils import log_utils
from utils.vision_utils.decorators import timeit
from vector_db.qdrant_context import VectorDbContext

# from vectoe_db.qdrant_context import VectorDbContext

logger = log_utils.LogUtils().get_logger(__name__)

rec = arc_face.ArcFace(os.path.abspath('../models/backbone_glint360k_r100.pth'), model_architecture="r100")
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


def find_all_images(base_path):
    # Use glob to find all .jpg files recursively
    for file_path in glob.glob(f"{base_path}/*.jpg", recursive=True):
        # Get the subfolder name relative to base_path
        name = os.path.basename(file_path).split('.')[0]
        yield name, file_path  # Yield one result at a time


# emb = load_embeddings(img_path='../samples/480.jpg')
# vector_db.upsert(vector=emb, name='Mehedi', id=1)
# vector_db.search_embedding(vector=emb)


index = 1
# base_path = 'D:/datasets/archive/lfw-funneled/lfw_funneled'
base_path = '/mnt/7A4CEE3F674E3964/ml_datasets/celeba/img_align_celeba/img_align_celeba'
# for name, img_path in find_images_one_by_one(base_path=base_path):
for name, img_path in find_all_images(base_path=base_path):
    # Generate a unique ID using UUID
    point_id = str(uuid.uuid4())
    logger.info(f'index: {index} id: {point_id} name: {name}')
    emb = load_embeddings(img_path=img_path)
    person = Person(person_id=index,
                    name=name,
                    email=f'{name}@example.com',
                    phone=name,
                    age=40,
                    city='New York',
                    country='USA',
                    address='USA'
                    )
    # print(person)
    # person.person_id = index
    # person.name = name
    vector_db.upsert_embedding(vector=emb, person=person)
    # vector_db.upsert(vector=emb, name=name, id=id)
    index += 1
