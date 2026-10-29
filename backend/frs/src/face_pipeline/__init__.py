"""
Reference implementation of the *client-side* face embedding pipeline.

The web (TypeScript) and mobile (Dart) clients implement exactly the same steps:

    5 landmarks -> similarity transform -> 112x112 aligned crop
    -> (x - 127.5) / 127.5, RGB, NCHW -> ArcFace model -> L2 normalised 512-d vector

This package is the source of truth used to generate the shared test vectors in
``shared/test-vectors`` and by the Python examples, so every platform produces
embeddings that live in the same vector space.
"""
from face_pipeline.alignment import ARCFACE_TEMPLATE_112, estimate_similarity_transform, alignment_matrix, warp_face
from face_pipeline.embedding import EMBEDDING_DIM, l2_normalize, cosine_similarity, fuse_embeddings
from face_pipeline.pose import HeadPose, pose_from_landmarks_3d, classify_pose
from face_pipeline.preprocess import to_model_input

__all__ = [
    "ARCFACE_TEMPLATE_112",
    "estimate_similarity_transform",
    "alignment_matrix",
    "warp_face",
    "EMBEDDING_DIM",
    "l2_normalize",
    "cosine_similarity",
    "fuse_embeddings",
    "HeadPose",
    "pose_from_landmarks_3d",
    "classify_pose",
    "to_model_input",
]
