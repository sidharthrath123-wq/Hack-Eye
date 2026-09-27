import os
import re
import ipaddress
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, status, Body
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBClassifier
import joblib
import pandas as pd
import tldextract
import uvicorn


# ============================================================
# APP INITIALIZATION & CORS SETUP
# ============================================================

app = FastAPI(
    title="CYBERGUARD URL & Technical Threat Detection API",
    description="FastAPI backend for ML-driven URL phishing detection and technical threat analysis.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class URLPredictRequest(BaseModel):
    url: str = Field(..., description="The URL to analyze for phishing threats")

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://example.com"
            }
        }
    }


class URLPredictResponse(BaseModel):
    url: str
    result: str
    risk_level: str
    phishing_probability: float
    legitimate_probability: float
    evidence: List[str]
    recommended_action: str


class TechnicalThreatResponse(BaseModel):
    classification: str
    normal_probability: float
    attack_probability: float
    risk_level: str
    explanation: List[str]
    recommended_action: str


class HealthResponse(BaseModel):
    system: str
    status: str


class ErrorResponse(BaseModel):
    error: str


# ============================================================
# CUSTOM VALIDATION EXCEPTION HANDLER
# ============================================================

@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        loc = first_error.get("loc", ())
        err_type = first_error.get("type", "")

        # Request body is completely missing or malformed JSON
        if loc == ("body",) or (len(loc) == 1 and loc[0] == "body") or err_type == "json_invalid":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Request body is empty."}
            )

        # Missing or invalid 'url' field
        if "url" in loc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "URL is required."}
            )

        field_name = loc[-1] if loc else "field"
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"{field_name} is required or invalid."}
        )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Request body is empty."}
    )


# ============================================================
# LOAD SAVED MODELS & ARTIFACTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "cyberguard_url_live_model.json")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "cyberguard_url_live_features.pkl")

model = XGBClassifier()
model.load_model(MODEL_PATH)

feature_columns = joblib.load(FEATURES_PATH)

print("CYBERGUARD URL MODEL LOADED SUCCESSFULLY!")
print("Number of features:", len(feature_columns))

# Technical Threat Model
TECHNICAL_MODEL_PATH = os.path.join(BASE_DIR, "model", "cyberguard_technical_model.json")
TECHNICAL_PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "cyberguard_technical_preprocessor.pkl")
TECHNICAL_FEATURE_INFO_PATH = os.path.join(BASE_DIR, "model", "cyberguard_technical_feature_info.pkl")

technical_model = XGBClassifier()
technical_model.load_model(TECHNICAL_MODEL_PATH)

technical_preprocessor = joblib.load(TECHNICAL_PREPROCESSOR_PATH)
technical_feature_info = joblib.load(TECHNICAL_FEATURE_INFO_PATH)

print("CYBERGUARD TECHNICAL THREAT MODEL LOADED SUCCESSFULLY!")
print("Technical raw features:", len(technical_feature_info["raw_features"]))


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url):
    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    host = parsed.hostname or ""

    # Basic features
    url_length = len(url)
    domain_length = len(host)

    # IP address
    try:
        ipaddress.ip_address(host)
        is_domain_ip = 1
    except:
        is_domain_ip = 0

    # Domain information
    extracted = tldextract.extract(url)

    subdomain = extracted.subdomain
    suffix = extracted.suffix

    tld_length = len(suffix)

    if subdomain:
        no_of_subdomain = len(
            [x for x in subdomain.split(".") if x]
        )
    else:
        no_of_subdomain = 0

    # Obfuscation
    encoded_chars = re.findall(
        r"%[0-9A-Fa-f]{2}",
        url
    )

    no_of_obfuscated_char = len(encoded_chars)

    has_obfuscation = (
        1 if no_of_obfuscated_char > 0 else 0
    )

    obfuscation_ratio = (
        no_of_obfuscated_char / url_length
        if url_length > 0 else 0
    )

    # Letters
    no_of_letters = len(
        re.findall(r"[A-Za-z]", url)
    )

    letter_ratio = (
        no_of_letters / url_length
        if url_length > 0 else 0
    )

    # Digits
    no_of_digits = len(
        re.findall(r"[0-9]", url)
    )

    digit_ratio = (
        no_of_digits / url_length
        if url_length > 0 else 0
    )

    # Special characters
    no_of_equals = url.count("=")
    no_of_question = url.count("?")
    no_of_ampersand = url.count("&")

    structural_chars = set("/.?=&")

    no_of_other_special_chars = sum(
        1
        for char in url
        if not char.isalnum()
        and char not in structural_chars
    )

    special_char_ratio = (
        no_of_other_special_chars / url_length
        if url_length > 0 else 0
    )

    # HTTPS
    is_https = (
        1
        if parsed.scheme.lower() == "https"
        else 0
    )

    # Create feature dictionary
    features = {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated_char,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_question,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special_chars,
        "SpacialCharRatioInURL": special_char_ratio,
        "IsHTTPS": is_https
    }

    return features


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(phishing_probability):
    if phishing_probability >= 0.80:
        return "CRITICAL"
    elif phishing_probability >= 0.60:
        return "HIGH"
    elif phishing_probability >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


# ============================================================
# EXPLANATION
# ============================================================

def generate_explanation(features, result):
    evidence = []

    if features["IsHTTPS"] == 0:
        evidence.append("The URL does not use HTTPS.")

    if features["IsDomainIP"] == 1:
        evidence.append(
            "The domain is an IP address instead of a normal domain name."
        )

    if features["HasObfuscation"] == 1:
        evidence.append(
            f"The URL contains {features['NoOfObfuscatedChar']} encoded or obfuscated character(s)."
        )

    if features["URLLength"] >= 75:
        evidence.append(
            f"The URL is unusually long ({features['URLLength']} characters)."
        )

    if features["NoOfSubDomain"] >= 3:
        evidence.append(
            f"The URL contains multiple subdomains ({features['NoOfSubDomain']})."
        )

    if features["NoOfDegitsInURL"] >= 5:
        evidence.append(
            f"The URL contains several digits ({features['NoOfDegitsInURL']})."
        )

    if features["NoOfQMarkInURL"] > 0:
        evidence.append("The URL contains query parameters.")

    if features["NoOfAmpersandInURL"] >= 2:
        evidence.append("The URL contains multiple URL parameters.")

    if features["NoOfOtherSpecialCharsInURL"] >= 3:
        evidence.append(
            f"The URL contains several special characters ({features['NoOfOtherSpecialCharsInURL']})."
        )

    # AI assessment
    if result == "PHISHING":
        evidence.append(
            "The XGBoost model detected a strong phishing pattern from the combined URL features."
        )
    else:
        evidence.append(
            "The XGBoost model did not detect a strong phishing pattern from the combined URL features."
        )

    return evidence


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def get_recommended_action(risk_level):
    if risk_level == "CRITICAL":
        return (
            "BLOCK URL and quarantine the request. Warn the user immediately."
        )
    elif risk_level == "HIGH":
        return (
            "BLOCK or strongly warn the user. Do not enter credentials or sensitive information."
        )
    elif risk_level == "MEDIUM":
        return (
            "WARN the user and recommend caution before opening the URL."
        )
    else:
        return (
            "ALLOW with caution. The URL does not show strong phishing indicators."
        )


# ============================================================
# URL ANALYSIS
# ============================================================

def analyze_url(url):
    # Extract features
    features = extract_url_features(url)

    # Create model input
    X_live = pd.DataFrame([features], columns=feature_columns)

    # Prediction
    prediction = model.predict(X_live)[0]
    probabilities = model.predict_proba(X_live)[0]

    phishing_probability = probabilities[0]
    legitimate_probability = probabilities[1]

    # Classification
    result = "PHISHING" if prediction == 0 else "LEGITIMATE"

    # Risk
    risk_level = get_risk_level(phishing_probability)

    # Explanation
    evidence = generate_explanation(features, result)

    # Recommended response
    recommended_action = get_recommended_action(risk_level)

    return {
        "url": url,
        "result": result,
        "risk_level": risk_level,
        "phishing_probability": round(float(phishing_probability), 6),
        "legitimate_probability": round(float(legitimate_probability), 6),
        "evidence": evidence,
        "recommended_action": recommended_action
    }


# ============================================================
# TECHNICAL THREAT ANALYSIS
# ============================================================

def analyze_technical_traffic(data):
    raw_features = technical_feature_info["raw_features"]
    input_data = {}

    for feature in raw_features:
        input_data[feature] = data.get(feature)

    X_raw = pd.DataFrame([input_data], columns=raw_features)

    # Convert numerical features
    numerical_features = technical_feature_info["numerical_features"]

    for feature in numerical_features:
        X_raw[feature] = pd.to_numeric(X_raw[feature], errors="coerce")

    # Check missing/invalid values
    if X_raw.isnull().any().any():
        missing = X_raw.columns[X_raw.isnull().any()].tolist()
        raise ValueError("Missing or invalid values for: " + ", ".join(missing))

    # Apply the saved preprocessing
    X_processed = technical_preprocessor.transform(X_raw)

    # Prediction
    prediction = technical_model.predict(X_processed)[0]
    probabilities = technical_model.predict_proba(X_processed)[0]

    normal_probability = probabilities[0]
    attack_probability = probabilities[1]

    # Classification
    result = "ATTACK" if prediction == 1 else "NORMAL"

    # Risk level
    if attack_probability >= 0.80:
        risk_level = "CRITICAL"
    elif attack_probability >= 0.60:
        risk_level = "HIGH"
    elif attack_probability >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Recommended action
    if risk_level == "CRITICAL":
        recommended_action = (
            "BLOCK/ISOLATE the suspicious traffic and alert the security analyst immediately."
        )
    elif risk_level == "HIGH":
        recommended_action = (
            "BLOCK or restrict the traffic and investigate the source immediately."
        )
    elif risk_level == "MEDIUM":
        recommended_action = (
            "MONITOR the traffic closely and investigate if suspicious behavior continues."
        )
    else:
        recommended_action = (
            "ALLOW the traffic but continue normal monitoring."
        )

    # Explanation
    if result == "ATTACK":
        explanation = [
            "The network traffic shows characteristics strongly associated with malicious activity.",
            "Multiple network-flow features increased the model's attack score.",
            "The model has high confidence that this traffic represents an attack."
        ]
    else:
        explanation = [
            "The network traffic does not show a strong attack pattern.",
            "Several network-flow features reduced the model's attack score.",
            "The model has high confidence that this traffic is normal."
        ]

    return {
        "classification": result,
        "normal_probability": round(float(normal_probability), 6),
        "attack_probability": round(float(attack_probability), 6),
        "risk_level": risk_level,
        "explanation": explanation,
        "recommended_action": recommended_action
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.post(
    "/predict",
    response_model=URLPredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Predict URL Phishing",
    description="Analyzes a given URL and determines if it is legitimate or phishing."
)
def predict(payload: URLPredictRequest):
    try:
        if not payload.url or not payload.url.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "URL is required."}
            )

        result = analyze_url(payload.url)
        return result

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )


@app.post(
    "/predict-technical",
    response_model=TechnicalThreatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Technical Threat Analysis",
    description="Analyzes network traffic flow data to classify attacks vs normal behavior."
)
def predict_technical(data: Optional[Dict[str, Any]] = Body(None)):
    try:
        if not data:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Request body is empty."}
            )

        result = analyze_technical_traffic(data)
        return result

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )


@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the status of the CYBERGUARD URL Phishing Detection API."
)
def home():
    return {
        "system": "CYBERGUARD URL Phishing Detection API",
        "status": "running"
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("====================================")
    print("     CYBERGUARD URL BACKEND")
    print("====================================")
    print("Server starting...")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )