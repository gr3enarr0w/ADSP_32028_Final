import os

# sentence-transformers is PyTorch-only; tell HuggingFace transformers to
# skip TensorFlow detection entirely. Without this, transformers finds a
# stale/incompatible TF install and imports tf_keras, which fails on
# macOS ARM and adds ~2 GB of unnecessary deps on Linux.
# Must be set before transformers is first imported (module-level init).
os.environ.setdefault("USE_TF", "0")
