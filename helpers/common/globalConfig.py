import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv('PROJECT_NAME')

DEFAULT_TIME_GAP = 1
DEFAULT_MACHINE_LEARNING_TRAIN_PROPORTION=0.75
VALIDATION_ITERATION = int(os.getenv('VALIDATION_ITERATION'))

DATASET_LOCATION = os.getenv('DATASET_LOCATION')
TEST_DATASET_LOCATION = os.getenv('TEST_DATASET_LOCATION')
CTU_DIR = os.getenv('CTU_DIR')
NCC_DIR = os.getenv('NCC_DIR')
NCC2_DIR = os.getenv('NCC2_DIR')
NCC_GRAPH_DIR = os.getenv('NCC_GRAPH_DIR')
OUT_DIR = os.getenv('OUT_DIR')

ABLATION = os.getenv('ABLATION')