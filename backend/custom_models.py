from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class IsoForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None):
        self.model = model

    def predict(self, X):
        if self.model is None:
            raise ValueError("❌ Internal model is missing. Retrain and save properly.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("❌ Internal model is missing. Retrain and save properly.")

        scores = self.model.decision_function(X)
        probs = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - probs, probs]).T

    def __setstate__(self, state):
        self.__dict__.update(state)

        # 🔥 CRITICAL FIX
        if getattr(self, "model", None) is None:
            print("⚠️ WARNING: IsoForestClassifier loaded WITHOUT internal model!")