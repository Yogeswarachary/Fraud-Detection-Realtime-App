from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
import sys
import __main__
import logging
import time

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr

from custom_models import IsoForestClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Register the custom model in __main__ so cloudpickle can unpickle it
__main__.IsoForestClassifier = IsoForestClassifier
sys.modules["__main__"].IsoForestClassifier = IsoForestClassifier


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


MODEL_FILES = {
    "catboost": MODELS_DIR / "catboost_model.pkl",
    "balanced_rf": MODELS_DIR / "balanced_random_forest_model_tuned.pkl",
    "hybrid_catboost": MODELS_DIR / "hybrid_catboost_model.pkl",
    "hybrid_lr": MODELS_DIR / "hybrid_logistic_regression_model.pkl",
}


app_state = {
    "models": {},
    "load_status": {},
    "feature_names": None,
    "thresholds": None,
    "explainers": {},
}


class TransactionType(str, Enum):
    CASH_OUT = "CASH_OUT"
    PAYMENT = "PAYMENT"
    CASH_IN = "CASH_IN"
    TRANSFER = "TRANSFER"
    DEBIT = "DEBIT"


class FraudInput(BaseModel):
    step: int = Field(..., ge=0, le=740)
    type: TransactionType
    amount: float = Field(..., ge=0.0, le=99999999.99)
    nameOrig: constr(min_length=10, max_length=10)
    oldbalanceOrg: float = Field(..., ge=0.0, le=99999999.99)
    newbalanceOrig: float = Field(..., ge=0.0, le=99999999.99)
    nameDest: constr(min_length=10, max_length=11)
    oldbalanceDest: float = Field(..., ge=0.0, le=99999999.99)
    newbalanceDest: float = Field(..., ge=0.0, le=99999999.99)


class ModelChoice(str, Enum):
    catboost = "catboost"
    balanced_rf = "balanced_rf"
    hybrid_catboost = "hybrid_catboost"
    hybrid_lr = "hybrid_lr"


def build_model_features(data: FraudInput, feature_names) -> pd.DataFrame:
    df = pd.DataFrame([data.model_dump()])
    df["type"] = df["type"].astype(str)

    df["hour"] = df["step"] % 24
    df["orig_balance_error"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    df["dest_balance_error"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    df["is_mule_orig"] = df["nameOrig"].str.startswith(("M", "C")).astype(int)
    df["is_mule_dest"] = df["nameDest"].str.startswith(("M", "C")).astype(int)

    df["transfer_to_zero_dest"] = (
        (df["type"] == "TRANSFER") & (df["oldbalanceDest"] == 0)
    ).astype(int)
    df["amount_to_orig_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    df["is_suspicious_pattern"] = (
        (df["type"].isin(["TRANSFER", "CASH_OUT"]))
        & (df["amount"] > 200000)
        & (df["oldbalanceOrg"] > 0)
    ).astype(int)

    numeric_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    for col in numeric_cols:
        df[f"{col}_log"] = np.log1p(df[col])

    upper = 2500000
    for col in numeric_cols:
        df[f"{col}_outlier"] = (df[col] > upper).astype(int)

    for transaction in ["CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT", "TRANSFER"]:
        df[f"type_{transaction}"] = (df["type"] == transaction).astype(int)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Base directory: {BASE_DIR}")

    loaded_models = {}
    load_status = {}

    for model_name, model_path in MODEL_FILES.items():
        try:
            logger.info(f"Loading {model_name} from {model_path}")
            with open(model_path, "rb") as f:
                loaded_models[model_name] = cloudpickle.load(f)
            load_status[model_name] = "loaded"
            logger.info(f"{model_name} loaded successfully")
        except Exception as e:
            load_status[model_name] = f"failed: {str(e)}"
            logger.error(f"Failed to load {model_name}: {e}")

    app_state["models"] = loaded_models
    app_state["load_status"] = load_status
    app_state["explainers"] = {}

    try:
        app_state["feature_names"] = joblib.load(MODELS_DIR / "fraud_features_names.pkl")
        logger.info("feature_names loaded successfully")
    except Exception as e:
        app_state["feature_names"] = None
        logger.error(f"Failed to load feature_names: {e}")

    try:
        app_state["thresholds"] = joblib.load(MODELS_DIR / "model_thresholds.pkl")
        logger.info("thresholds loaded successfully")
    except Exception as e:
        app_state["thresholds"] = None
        logger.error(f"Failed to load thresholds: {e}")

    try:
        if "catboost" in loaded_models:
            app_state["explainers"]["catboost"] = shap.TreeExplainer(loaded_models["catboost"])
            logger.info("CatBoost explainer loaded successfully")
    except Exception as e:
        logger.warning(f"Explainer load failed: {e}")

    yield
    app_state.clear()


app = FastAPI(
    title="JPMC Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "message": "Validation failed",
            "errors": exc.errors(),
            "body": exc.body,
        },
    )


@app.get("/")
async def root():
    return {
        "message": "JPMC Fraud Detection FastAPI is running",
        "available_models": list(app_state.get("models", {}).keys()),
    }


@app.get("/health")
async def health():
    return {
        "loaded_models": list(app_state.get("models", {}).keys()),
        "load_status": app_state.get("load_status", {}),
        "feature_names_loaded": app_state.get("feature_names") is not None,
        "thresholds_loaded": app_state.get("thresholds") is not None,
    }


@app.get("/sample-input")
async def sample_input():
    return {
        "step": 740,
        "type": "TRANSFER",
        "amount": 99999999.99,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 99999999.99,
        "newbalanceOrig": 99999999.99,
        "nameDest": "M1234567890",
        "oldbalanceDest": 99999999.99,
        "newbalanceDest": 99999999.99,
    }


@app.get("/debug/features")
async def debug_features():
    feature_names = app_state.get("feature_names", [])
    return {
        "feature_count": len(feature_names) if feature_names else 0,
        "feature_names": feature_names,
        "loaded_models": list(app_state.get("models", {}).keys()),
        "load_status": app_state.get("load_status", {}),
    }


@app.get("/debug/model-info")
async def debug_model_info():
    info = {}
    for name, model in app_state.get("models", {}).items():
        info[name] = {
            "type": str(type(model)),
            "has_predict_proba": hasattr(model, "predict_proba"),
        }
    return info


@app.post("/predict/{model_choice}")
def predict(model_choice: ModelChoice, data: FraudInput):
    start_time = time.time()
    model_key = model_choice.value

    logger.info(f"Incoming request for model: {model_key}")
    logger.info(f"Input data: {data.model_dump()}")

    model = app_state["models"].get(model_key)

    if model is None:
        logger.warning(f"{model_key} not available. Falling back to CatBoost")
        model = app_state["models"].get("catboost")
        model_key = "catboost"

    feature_names = app_state["feature_names"]
    threshold = app_state["thresholds"].get(model_key, 0.5)

    input_df = build_model_features(data, feature_names)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    logger.info(f"Prediction: {prediction}, Probability: {probability}")

    top_reasons = []

    if probability > threshold:
        try:
            if model_key == "catboost":
                explainer = app_state.get("explainers", {}).get(model_key)

                if explainer is None:
                    explainer = shap.TreeExplainer(model)
                    app_state["explainers"][model_key] = explainer

                shap_values = explainer.shap_values(input_df)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                shap_values = shap_values[0]

                feature_impact = list(zip(feature_names, shap_values))
                feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

                top_reasons = [
                    {"feature": f, "impact": float(v)}
                    for f, v in feature_impact[:3]
                ]
            else:
                raise Exception("Use fallback for non-catboost models")

        except Exception as e:
            logger.warning(f"SHAP error: {e}")
            top_reasons = []

    decision = "FRAUD" if probability >= threshold else "SAFE"
    response_time = round(time.time() - start_time, 4)

    return {
        "model_used": model_key,
        "input_received": data.model_dump(),
        "prediction": int(prediction),
        "fraud_risk_score": float(probability),
        "decision": decision,
        "threshold_used": threshold,
        "top_reasons": top_reasons,
        "response_time": response_time,
    }