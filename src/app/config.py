import os

ORIGINS = os.environ.get(
    "ORIGINS",
    "http://localhost:8080",
).split(" ")
0

DEBUG = os.environ.get("DEBUG")

DATA_PATH = os.environ.get("DATA_PATH", ".")
DB_PATH = os.environ.get("DB_PATH", "FTS.sqlite")

X_ENDPOINT = os.environ.get("X_ENDPOINT")
