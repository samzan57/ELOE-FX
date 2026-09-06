# models/lgbm_model.py
import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from features.technical import FEATURE_COLS


# ─────────────────────────────────────────────
#  Hyperparamètres optimisés pour séries temporelles financières
#  (shallow trees + forte régularisation = généralisation > in-sample fit)
# ─────────────────────────────────────────────
_LGBM_PARAMS = dict(
    objective        = "binary",
    metric           = "binary_logloss",
    n_estimators     = 600,
    num_leaves       = 24,          # peu de feuilles → moins d'overfitting
    max_depth        = 5,
    learning_rate    = 0.03,
    subsample        = 0.75,
    colsample_bytree = 0.65,
    min_child_samples= 60,          # crucial : interdit d'apprendre sur < 60 trades
    reg_alpha        = 0.3,
    reg_lambda       = 2.0,
    class_weight     = "balanced",
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1,
)


class LGBMTradingModel:
    """
    Wrapper LightGBM pour signaux de trading directionnels.
    Inclut : early stopping, threshold EV-optimal, feature importance.
    """

    def __init__(self, side: str = "buy"):
        assert side in ("buy", "sell")
        self.side      = side
        self.model     = None
        self.threshold = 0.50
        self.features  = FEATURE_COLS
        self._y_col    = "y_buy" if side == "buy" else "y_sell"

    # ── Entraînement ─────────────────────────────
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        if not _HAS_LGB:
            raise ImportError("lightgbm non installé : pip install lightgbm")

        tr = train_df.dropna(subset=self.features + [self._y_col])
        X_tr = tr[self.features].values
        y_tr = tr[self._y_col].values

        callbacks = [lgb.log_evaluation(period=0)]

        if val_df is not None:
            vl = val_df.dropna(subset=self.features + [self._y_col])
            X_vl, y_vl = vl[self.features].values, vl[self._y_col].values
            callbacks.append(lgb.early_stopping(stopping_rounds=60, verbose=False))
            self.model = lgb.LGBMClassifier(**_LGBM_PARAMS)
            self.model.fit(X_tr, y_tr,
                           eval_set=[(X_vl, y_vl)],
                           callbacks=callbacks)
        else:
            self.model = lgb.LGBMClassifier(**_LGBM_PARAMS)
            self.model.fit(X_tr, y_tr, callbacks=callbacks)

        return self

    # ── Prédiction ───────────────────────────────
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].fillna(0).values
        return self.model.predict_proba(X)[:, 1]

    # ── Threshold EV-optimal ──────────────────────
    def set_ev_threshold(self, rr: float, safety_margin: float = 0.025):
        """
        Seuil théorique : p* = 1/(1+rr).
        On ajoute une marge de sécurité pour ne prendre que les setups à EV clairement positif.
        """
        self.threshold = round(1.0 / (1.0 + rr) + safety_margin, 4)

    # ── Feature importance ────────────────────────
    def feature_importance(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    # ── Persistance ──────────────────────────────
    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "LGBMTradingModel":
        return joblib.load(path)
