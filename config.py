"""
Simple configuration file for brain MRI classification training
"""
import os 
import argparse

parser = argparse.ArgumentParser(description='run training')
parser.add_argument('-c', '--column', type=str,  default=None, help='Select one  target column for trainig')
parser.add_argument('-m', '--mode', type=str,  default=None, help='Select one of sfcn, dense, linear, ssl-finetuned, lora')
parser.add_argument('-g', '--gpu', type=str,  default=None, help='Select one of sfcn, dense, linear, ssl-finetuned, lora')
parser.add_argument('-i', '--eid', type=str, nargs='*', default=None, help='Create heatmaps for selected eid')
args = parser.parse_args()

EIDS = args.eid

# ============================================================================
# BASIC SETTINGS
# ============================================================================
COLUMN_NAME = 'last_progression_pst_15z'
CSV_NAME = 'last_progression_pst_15z'
TRAINING_MODE = 'linear'  # Options: 'sfcn', 'dense', 'linear', 'ssl-finetuned', 'lora'
TASK = 'classification'

if not args.column is None:
    COLUMN_NAME = args.column
    CSV_NAME = args.column

if not args.mode is None:
    TRAINING_MODE = args.mode

# ============================================================================
# DATA PATHS
# ============================================================================

TRAIN_COHORT = 'mspaths/flair'
TEST_COHORT = 'mspaths2/flair'

CSV_TRAIN = f'../../data/{TRAIN_COHORT}/train/{CSV_NAME}.csv'
CSV_VAL = f'../../data/{TRAIN_COHORT}/val/{CSV_NAME}.csv'
CSV_TEST = f'../../data/{TEST_COHORT}/test/{CSV_NAME}.csv'

TENSOR_DIR = f'../../images/{TRAIN_COHORT}/'
TENSOR_DIR_TEST = f'../../images/{TEST_COHORT}/'

# ============================================================================
# MODEL SETTINGS
# ============================================================================
IMG_SIZE = 96
N_CHANNELS = 1
N_CLASSES = 2

# LoRA Parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ['feature_extractor.conv_']

# SSL Pretrained Model
SSL_COHORT = 'ukb-nako'
SSL_BATCH_SIZE = 16
SSL_EPOCHS = 1000
PRETRAINED_MODEL = (f'../../models/ssl/sfcn/{SSL_COHORT}/'
                   f'{SSL_COHORT}{IMG_SIZE}/final_model_b{SSL_BATCH_SIZE}_e{SSL_EPOCHS}.pt')


# ============================================================================
# TRAINING SETTINGS
# ============================================================================
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 0.1
NUM_WORKERS = 8
DEVICE = "cuda:0"

if not args.gpu is None:
    DEVICE = args.gpu

SEED = 42
NROWS = None  # Set to None to use all data, or int for subset

# Early Stopping
PATIENCE = 20

# Learning Rate Scheduler
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

# ============================================================================
# OUTPUT PATHS
# ============================================================================
# Experiment name
EXPERIMENT_NAME = f"{CSV_NAME}_e{NUM_EPOCHS}_b{BATCH_SIZE}_im{IMG_SIZE}"

# Output directories
MODEL_DIR = f'../../models/'
SCORES_DIR = f'../../scores'
LOG_DIR = f'../../logs'
EVALUATION_DIR = f'../../evaluations/'
EXPLAINABILITY_DIR = f"../../explainability/"
  
# ============================================================================
# HEATMAP CONFIGURATION 
# ============================================================================
HEATMAP_MODE = 'top_individual'  # Options: 'single', 'average', 'top_individual'
HEATMAP_TOP_N = 5
ATTENTION_METHOD = 'saliency'  # Options: 'saliency', 'gradcam'
ATTENTION_MODE = 'magnitude'  # Options: 'magnitude', 'signed'
ATTENTION_TARGET = 'logit_diff'  # Options: 'logit_diff', 'pred', 'target_class'
ATTENTION_CLASS_IDX = None
ATLAS_PATH = 'atlas_resampled_96.nii.gz'



  
