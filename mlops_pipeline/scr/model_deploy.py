from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Cargar modelo
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()