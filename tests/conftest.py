"""Fix FAISS + PyTorch OMP conflict on macOS."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
