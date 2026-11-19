"""Video indexing pipeline: parquet rows -> decoded frames -> YOLO11l -> per-second lines."""
import os

# Notebook kernels (Colab/Jupyter) export MPLBACKEND=module://matplotlib_inline...; subprocesses
# inherit it, and matplotlib (imported by Ultralytics) rejects it outside the kernel. We never plot.
if os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"
