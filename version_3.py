import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


BASE_DIR = "YOUR_BASE_DIR"
TRAIN_PATH  = os.path.join(BASE_DIR, "train.csv")
TEST_PATH   = os.path.join(BASE_DIR, "test.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "submission_porto_mlp_lgb_fusion.csv")

N_FOLDS      = 5
SEED         = 42

MLP_BATCH_SIZE   = 2048
MLP_EPOCHS       = 40
MLP_PATIENCE     = 8
MLP_LR           = 1e-3
MLP_WEIGHT_DECAY = 1e-4

N_QUANTILES  = 100_000
SUBSAMPLE    = 200_000
GAUSS_THRESH = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


train = pd.read_csv(TRAIN_PATH, na_values=-1)
test  = pd.read_csv(TEST_PATH,  na_values=-1)

y        = train["target"].values.astype(np.float32)
test_ids = test["id"].values

train = train.drop(columns=["id", "target"])
test  = test.drop(columns=["id"])

drop_calc = [c for c in train.columns if c.startswith("ps_calc_")]
drop_extra = [
    "ps_ind_14",
    "ps_ind_10_bin", "ps_ind_11_bin", "ps_ind_12_bin", "ps_ind_13_bin",
    "ps_car_10_cat", "ps_car_11", "ps_car_12", "ps_car_14",
]
drop_cols = [c for c in drop_calc + drop_extra if c in train.columns]
train = train.drop(columns=drop_cols)
test  = test.drop(columns=drop_cols)

low_card_cat = [c for c in train.columns if c.endswith("_cat")]

low_unique_num = [
    "ps_car_15", "ps_ind_01", "ps_ind_03",
    "ps_ind_15", "ps_reg_01", "ps_reg_02",
]
low_unique_num = [c for c in low_unique_num if c in train.columns]

num_cols = [
    c for c in train.columns
    if c not in low_card_cat + low_unique_num
]

high_card_num = [c for c in num_cols if train[c].nunique() > GAUSS_THRESH]
low_card_num  = [c for c in num_cols if c not in high_card_num]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  make_ohe()),
])

low_unique_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("onehot",  make_ohe()),
])

gauss_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("gaussrank", QuantileTransformer(
        n_quantiles=N_QUANTILES,
        output_distribution="normal",
        subsample=SUBSAMPLE,
        random_state=SEED,
    )),
])

scale_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("cat",        cat_pipe,        low_card_cat),
    ("low_unique", low_unique_pipe, low_unique_num),
    ("high_num",   gauss_pipe,      high_card_num),
    ("low_num",    scale_pipe,      low_card_num + low_unique_num),
], remainder="drop")

all_X = pd.concat([train, test], axis=0, ignore_index=True)
X_all = preprocessor.fit_transform(all_X).astype(np.float32)
n_train = len(train)

X_train = X_all[:n_train]
X_test  = X_all[n_train:]

X_train_tensor = torch.tensor(X_train, device=DEVICE, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test,  device=DEVICE, dtype=torch.float32)
y_tensor       = torch.tensor(y,       device=DEVICE, dtype=torch.float32)


class PortoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def iterate_batches(X, y, batch_size, shuffle=True):
    n = X.size(0)
    perm = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        if idx.numel() < 2:
            continue
        yield X[idx], y[idx]


def train_one_epoch(model, X, y, optimizer, criterion, batch_size):
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=True):
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def predict_probs(model, X, batch_size=8192):
    model.eval()
    probs = []
    for i in range(0, X.size(0), batch_size):
        xb = X[i:i + batch_size]
        probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def evaluate_auc(model, X, y):
    probs = predict_probs(model, X)
    return roc_auc_score(y.cpu().numpy(), probs)


pos_weight = torch.tensor(
    [(y == 0).sum() / (y == 1).sum()],
    dtype=torch.float32, device=DEVICE,
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "feature_fraction": 0.9,
    "min_data_in_leaf": 24,
    "lambda_l1": 10,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "learning_rate": 0.01,
    "num_leaves": 24,
    "verbose": -1,
    "seed": SEED,
    "num_threads": os.cpu_count() or 4,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

test_mlp = np.zeros(len(test_ids))
test_lgb = np.zeros(len(test_ids))

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
    tr_idx_t = torch.tensor(tr_idx, device=DEVICE)
    va_idx_t = torch.tensor(va_idx, device=DEVICE)

    X_tr, y_tr = X_train_tensor[tr_idx_t], y_tensor[tr_idx_t]
    X_va, y_va = X_train_tensor[va_idx_t], y_tensor[va_idx_t]

    mlp = PortoMLP(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_auc, best_state, wait = 0.0, None, 0

    for epoch in range(1, MLP_EPOCHS + 1):
        loss = train_one_epoch(mlp, X_tr, y_tr, optimizer, criterion, MLP_BATCH_SIZE)
        val_auc = evaluate_auc(mlp, X_va, y_va)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= MLP_PATIENCE:
            break

    mlp.load_state_dict(best_state)
    test_mlp += predict_probs(mlp, X_test_tensor) / N_FOLDS

    X_tr_lgb, y_tr_lgb = X_train[tr_idx], y[tr_idx]
    X_va_lgb, y_va_lgb = X_train[va_idx], y[va_idx]

    lgb_train = lgb.Dataset(X_tr_lgb, y_tr_lgb)
    lgb_valid = lgb.Dataset(X_va_lgb, y_va_lgb, reference=lgb_train)

    lgb_model = lgb.train(
        LGB_PARAMS,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    test_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / N_FOLDS


best_w = 0.5
final_test = best_w * test_mlp + (1 - best_w) * test_lgb

sample_sub = pd.read_csv(SAMPLE_PATH)
pred_map = pd.Series(final_test, index=test_ids)
sample_sub["target"] = sample_sub["id"].map(pred_map)
sample_sub.to_csv(OUTPUT_PATH, index=False)