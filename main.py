import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. DEFINE THE INPUT SCHEMA (PYDANTIC)
# ---------------------------------------------------------
# This schema maps exactly to the numeric and categorical features
# we defined in the training script.

class SBALoanApplication(BaseModel):
    # Numeric Features
    Term: int = Field(..., gt=0, description="Loan term in months")
    NoEmp: int = Field(..., ge=0, description="Number of employees")
    CreateJob: int = Field(..., ge=0, description="Number of jobs created")
    RetainedJob: int = Field(..., ge=0, description="Number of jobs retained")
    GrAppv: float = Field(..., gt=0, description="Gross approved loan amount ($)")
    Guarantee_Ratio: float = Field(..., ge=0.0, le=1.0, description="Percentage of loan guaranteed by SBA")
    
    # Categorical Features (We type them as strings/ints to match the pipeline's expectations)
    NAICS_Sector: str = Field(..., description="First 2 digits of NAICS code (e.g., '72' for Accommodation/Food)")
    NewExist: str = Field(..., description="'1' for Existing business, '2' for New business")
    UrbanRural: str = Field(..., description="'1' for Urban, '2' for Rural, '0' for Undefined")
    IsFranchise: str = Field(..., description="'0' for No, '1' for Yes")
    RealEstate: str = Field(..., description="'1' if backed by real estate (Term >= 240), '0' otherwise")
    RevLineCr: str = Field(..., description="Revolving line of credit? 'Y' or 'N'")
    LowDoc: str = Field(..., description="LowDoc Loan Program? 'Y' or 'N'")

    class Config:
        # This populates the Swagger UI (/docs) with a realistic test payload
        json_schema_extra = {
            "example": {
                "Term": 84,
                "NoEmp": 4,
                "CreateJob": 0,
                "RetainedJob": 4,
                "GrAppv": 150000.0,
                "Guarantee_Ratio": 0.5,
                "NAICS_Sector": "72",
                "NewExist": "1",
                "UrbanRural": "1",
                "IsFranchise": "0",
                "RealEstate": "0",
                "RevLineCr": "N",
                "LowDoc": "N"
            }
        }

# ---------------------------------------------------------
# 2. MODEL LIFECYCLE MANAGEMENT
# ---------------------------------------------------------
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the serialized pipeline into memory exactly once.
    try:
        # This loads the ColumnTransformer AND the XGBoost model together
        ml_models["pipeline"] = joblib.load("sba_pipeline.joblib")
        print("SBA Pipeline loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model artifact: {e}")
    
    yield # The API serves requests while yielding here
    
    # Shutdown: Clean up resources
    ml_models.clear()

# Initialize the API
app = FastAPI(
    title="SBA Commercial Lending Decision API",
    description="Real-time default prediction engine for SBA 7(a) small business loans.",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# 3. DEFINE THE ENDPOINTS
# ---------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    """Confirms the API is alive and the model artifact is loaded in memory."""
    if "pipeline" not in ml_models:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    return {"status": "healthy", "model": "sba_pipeline.joblib"}

@app.post("/predict", tags=["Decision Engine"])
async def predict_default(application: SBALoanApplication):
    """
    Accepts a validated JSON application, executes the Scikit-Learn preprocessing pipeline,
    scores it via XGBoost, and returns a credit decision.
    """
    try:
        # 1. Convert the validated Pydantic object to a dictionary
        app_dict = application.model_dump()
        
        # 2. Convert to a single-row Pandas DataFrame 
        # The column names perfectly match what the pipeline expects
        df = pd.DataFrame([app_dict])
        
        # 3. Retrieve the pipeline from memory
        pipeline = ml_models["pipeline"]
        
        # 4. Predict probability of default (Class 1)
        # The pipeline handles scaling, imputing, and one-hot encoding under the hood!
        pd_score = pipeline.predict_proba(df)[0][1]
        
        # 5. Apply a rudimentary risk policy
        # If the probability of default is greater than 35%, we decline.
        decision = "APPROVE" if pd_score < 0.35 else "DECLINE"
        
        return {
            "probability_of_default": round(float(pd_score), 4),
            "risk_tier": "High" if pd_score > 0.25 else "Low",
            "decision": decision
        }
        
    except Exception as e:
        # In a real system, you would log the traceback here.
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")