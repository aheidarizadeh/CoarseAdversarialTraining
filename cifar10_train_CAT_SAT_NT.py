import importlib
import cifar10_utils 
importlib.reload(cifar10_utils)
from cifar10_utils import *

# ---- GPU setup (safe defaults) ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter logs

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')  # pick first GPU
        print("[INFO] Using GPU:", tf.config.get_visible_devices('GPU'))
    except Exception as e:
        print("[WARN] Could not set memory growth/visibility:", e)
else:
    print("[WARN] No GPU detected—running on CPU")

# ---- Config ----
BATCH_SIZE = 64
EPOCHS = 32
EPS = 8/255.0
ALPHA = 2/255.0
PGD_STEPS_TRAIN = 10
PGD_STEPS_EVAL = 20
LR = 5e-4

# Load CIFAR-10
X_train, y_train_f, X_test, y_test_f, train_ds, test_ds, test_labels = load_cifar10()

# Define mapping

# ------------- Mapping  -------------
T1 = [[0,1,8,9], [3,4,5,7], [2,6]] # T1(vehicles/mammals/other)
#T2 = [[0,8], [1,9], [2,3,4,5,6,7]] #T2 (ship and airplane/truck and car/all animalls)
#T3 = [[0,8], [1,9], [3,4,5,7], [2,6]] #T3 ship and airplane/truck and car/mammals/other

# Random grouping into 3 clusters
#T4 = [[0, 3, 6], [1, 4, 8], [2, 5, 7, 9]] #T4 Random
#T5 = [[0, 5], [1, 7, 9], [2, 8], [3, 4, 6]] #T5 Random

Mapping_T = T1
A_np, A_tf, coarse_test_labels = build_mapping(Mapping_T, test_labels)

if __name__ == "__main__":
    
    # Train NT model
    train_nt_cifar10(train_ds, run_tag="NT32ep", epochs=EPOCHS, save_root="./CIFAR10_models")


    sat_model =train_sat(train_ds, eps=EPS, alpha=ALPHA, steps=PGD_STEPS_TRAIN, 
                    pretrained_path=None,
                    out_dir="./CIFAR10_models",
                    attack = "PGD")

    cat_model = train_cat(train_ds, Mapping_T, A_tf, eps=EPS, alpha=ALPHA, steps=PGD_STEPS_TRAIN, 
                    pretrained_path=None,
                    out_dir="./CIFAR10_models",
                    tag="T1", # T1 mapping selected
                    attack = "PGD")