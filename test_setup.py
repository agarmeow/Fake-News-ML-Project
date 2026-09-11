import numpy
import pandas
import sklearn
import torch
import transformers

print("===== FAKE NEWS ML PROJECT =====")

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

print("\n===== GPU CHECK =====")
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

print("\n===== SETUP SUCCESSFUL =====")