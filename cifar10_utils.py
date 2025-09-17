import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model , models

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

# ---------------- Default Config ----------------
BATCH_SIZE = 64
EPOCHS = 16
EPS = 8 / 255.0
ALPHA = 2 / 255.0
EPS_EVAL = 8 / 255.0
PGD_STEPS_TRAIN = 10          # training attack steps
PGD_STEPS_EVAL = 20     # eval attack steps
LR = 5e-4

# ---------------- Data Loader------------------
def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    X_train = train_images.astype(np.float32) / 255.0
    X_test = test_images.astype(np.float32) / 255.0

    y_train_f = to_categorical(train_labels, 10).astype(np.float32)
    y_test_f = to_categorical(test_labels, 10).astype(np.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train_f))
        .shuffle(50000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test_f))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return X_train, y_train_f, X_test, y_test_f, train_ds, test_ds , test_labels

# ---------------- Mapping Utils ----------------
def build_mapping(mapping_T, test_labels):
    M, Mc = 10, len(mapping_T)
    A_np = np.zeros((M, Mc), dtype=np.float32)
    for j, group in enumerate(mapping_T):
        for m in group:
            A_np[m, j] = 1.0
    A_tf = tf.constant(A_np)

    # Coarse labels for test set (ints)
    coarse_test_labels = np.zeros((len(test_labels),), dtype=np.int64)
    for i in range(len(test_labels)):
        fine = int(test_labels[i, 0])
        coarse_test_labels[i] = [jj for jj, g in enumerate(mapping_T) if fine in g][0]

    return A_np, A_tf, coarse_test_labels



# ------------- PGD (works for SAT & CAT) -------------
ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


def pgd_linf(model, x, y_fine, y_coarse, eps, alpha, steps, loss_mode, A_tf,
             training_flag=False, random_start=True):
    """
    model: outputs fine softmax p_f (B,10)
    loss_mode: 'fine' (SAT) or 'coarse' (CAT attack) or 'joint'
    random_start: if True, start from x + U(-eps, eps) (FGSM_r / PGD-r); if False, start at x (vanilla FGSM)
    """
    x0 = tf.identity(x)

    if random_start:
        x_adv = x0 + tf.random.uniform(tf.shape(x0), -eps, eps, dtype=x0.dtype)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    else:
        x_adv = tf.identity(x0)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            p_f = model(x_adv, training=training_flag)
            if loss_mode == "coarse": #CAT 
                q = tf.matmul(p_f, A_tf)
                loss = ce(y_coarse, q)
            else:   #SAT 
                loss = ce(y_fine, p_f)

        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        # project to l_inf ball & valid pixel range
        x_adv = tf.clip_by_value(tf.minimum(tf.maximum(x_adv, x0 - eps), x0 + eps), 0.0, 1.0)

    return tf.stop_gradient(x_adv)
# ------------- Eval helpers -------------
@tf.function(jit_compile=False)
def _fine_ce(y_true, p_f):
    # small wrapper for tf.function friendliness
    return ce(y_true, p_f)



def build_vgg_like(input_shape=(32, 32, 3), num_classes=10):
    """Reconstruct the CIFAR10_VGG16LIKE (~1.2M params) model."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3,3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),

        # Block 2
        layers.Conv2D(128, (3,3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),

        # Block 3
        layers.Conv2D(128, (3,3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax", name="SM_fine"),
    ])
    return model

def train_nt_cifar10(train_ds, run_tag="NT_clean", epochs=EPOCHS, save_root="./CIFAR10_models"):
    """
    Natural Training (NT) for CIFAR-10 using clean data only.
    - train_ds: tf.data.Dataset from load_cifar10()
    - X_test, y_test_f: test images + one-hot labels
    - test_labels: fine test labels (needed for CASR eval)
    - mapping_T: coarse mapping
    """
    os.makedirs(save_root, exist_ok=True)

    # Build model
    model = build_vgg_like()
    opt   = tf.keras.optimizers.Adam(LR)

    for ep in range(1, epochs+1):
        losses = []
        for xb, yb in train_ds:
            with tf.GradientTape() as tape:
                p = model(xb, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(yb, p)
                )
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss.numpy()))

        print(f"[NT-CIFAR] Epoch {ep:02d}/{epochs} | "
                f"train_loss={np.mean(losses):.4f} | ")

    # Save model
    final_path = os.path.join(save_root, f"CIFAR10_{run_tag}_final.keras")
    model.save(final_path)
    print(f"[NT-CIFAR] Saved -> {final_path}")



def train_sat(train_ds, eps, alpha, steps, pretrained_path=None,
                    out_dir="./CIFAR10_models",
                    attack = "PGD"):
    # Choose model source
    if pretrained_path is None:
        print("[SAT] No pretrained path provided → building from scratch")
        model = build_vgg_like()
    else:
        print(f"[SAT] Loading pretrained model from {pretrained_path}")
         model = load_model(pretrained_path, compile=False)
    opt = tf.keras.optimizers.Adam(LR)

    for ep in range(EPOCHS):
        running = 0.0
        nsteps = 0
        for xb, yb in train_ds:
            # PGD on fine loss
            x_adv = pgd_linf(
                model, xb, yb, None, eps, alpha, steps,
                "fine", A_tf=None, training_flag=False
            )
            with tf.GradientTape() as tape:
                p_f = model(x_adv, training=True)
                loss = _fine_ce(yb, p_f)
            vars_ = model.trainable_variables
            grads = tape.gradient(loss, vars_)
            grads_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]  # prevents warnings
            opt.apply_gradients(grads_vars)
            running += float(loss.numpy())
            nsteps += 1

        print(f"[SAT] Epoch {ep + 1:02d}/{EPOCHS} | adv-train-loss={running / nsteps:.4f}")


    final = os.path.join(out_dir, f"CIFAR10_VGG16LIKE_SAT_{attack}{PGD_STEPS_TRAIN}_eps8_final.keras")
    model.save(final)
    print(f"[SAT] Saved -> {final}")
    return model


def train_cat(train_ds, mapping_T, A_tf, eps, alpha, steps, pretrained_path=None,
                    out_dir="./CIFAR10_models",
                    tag="T1",
                    attack = "PGD"):
        # Choose model source
    if pretrained_path is None:
        print("[CAT] No pretrained path provided → building from scratch")
        model = build_vgg_like()
    else:
        print(f"[CAT] Loading pretrained model from {pretrained_path}")
         model = load_model(pretrained_path, compile=False)
    Mc = len(mapping_T)    
    opt = tf.keras.optimizers.Adam(LR)
    for ep in range(EPOCHS):
        running = 0.0
        nsteps = 0
        for xb, yb in train_ds:
            # build coarse one-hots for this batch
            fine_idx = yb.numpy().argmax(1)
            batch_coarse = np.zeros((xb.shape[0], Mc), dtype=np.float32)
            for i, f in enumerate(fine_idx):
                cj = [jj for jj, g in enumerate(mapping_T) if f in g][0]
                batch_coarse[i, cj] = 1.0
            yb_c = tf.constant(batch_coarse, dtype=tf.float32)

            # PGD on COARSE loss
            x_adv = pgd_linf(
                model, xb, yb, yb_c, eps, alpha, steps,
                "coarse", A_tf=A_tf, training_flag=False
            )

            with tf.GradientTape() as tape:
                p_f = model(x_adv, training=True)
                q = tf.matmul(p_f, A_tf)
                loss = ce(yb_c, q)   # coarse-only CE
            vars_ = model.trainable_variables
            grads = tape.gradient(loss, vars_)
            grads_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
            opt.apply_gradients(grads_vars)
            running += float(loss.numpy())
            nsteps += 1

        print(f"[CAT] Epoch {ep + 1:02d}/{EPOCHS} | adv-train-loss={running / nsteps:.4f}")

    final = os.path.join(out_dir, f"CIFAR10_VGG16LIKE_CAT_{attack}{PGD_STEPS_TRAIN}_eps8_{tag}_final.keras")
    model.save(final)
    print(f"[CAT] Saved -> {final}")

    return model

def eval_casr(model, test_ds, test_labels, mapping_T, eps=EPS_EVAL, alpha=ALPHA, steps=PGD_STEPS_EVAL):
    """CASR(ε): fraction of coarse label flips under a coarse attack on q = p_f @ A."""
    flips = total = 0
    Mc = len(mapping_T)
    A_np, A_tf, coarse_test_labels = build_mapping(mapping_T, test_labels)
    for xb, yb in test_ds:
        # build coarse one-hots for this batch
        fine_idx = yb.numpy().argmax(1)
        batch_coarse = np.zeros((xb.shape[0], Mc), dtype=np.float32)
        for i, f in enumerate(fine_idx):
            cj = [jj for jj, g in enumerate(mapping_T) if f in g][0]
            batch_coarse[i, cj] = 1.0
        yb_c = tf.constant(batch_coarse, dtype=tf.float32)

        x_adv = pgd_linf(
            model, xb, yb, yb_c, eps, alpha, steps,
            loss_mode="coarse", A_tf=A_tf, training_flag=False, random_start=True
        )
        p_f = model(x_adv, training=False).numpy()
        q = p_f @ A_np
        coarse_pred = q.argmax(1)

        # compare against precomputed coarse_test_labels slice
        s = total
        e = total + xb.shape[0]
        flips += int(np.sum(coarse_pred != coarse_test_labels[s:e]))
        total += xb.shape[0]
    return flips / total



