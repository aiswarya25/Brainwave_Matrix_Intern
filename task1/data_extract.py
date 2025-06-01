
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import os

zip_path = "archive (3).zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("dataset")

os.listdir("dataset")
