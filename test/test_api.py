import requests
import json

# 1. Replace this with your actual Google Cloud Run URL
# IMPORTANT: Make sure to keep the "/predict" at the end of the URL
CLOUD_RUN_URL = "https://sba-decision-api-872178421051.asia-south1.run.app/predict"

# 2. Define the "Toxic Loan" payload
# Short term, new business, restaurant sector, no real estate backing
payload = {
    "Term": 12,
    "NoEmp": 1,
    "CreateJob": 0,
    "RetainedJob": 1,
    "GrAppv": 25000.0,
    "Guarantee_Ratio": 0.5,
    "NAICS_Sector": "72",
    "NewExist": "2",
    "UrbanRural": "1",
    "IsFranchise": "0",
    "RealEstate": "0",
    "RevLineCr": "Y",
    "LowDoc": "N"
}

print(f"Sending request to: {CLOUD_RUN_URL}...")

try:
    # 3. Send the POST request to the live API
    response = requests.post(
        CLOUD_RUN_URL, 
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10 # Best practice: don't let the script hang forever
    )
    
    # 4. Check if the server returned a 200 OK status
    response.raise_for_status()
    
    # 5. Parse and print the prediction
    result = response.json()
    print("\n✅ API Response Received:")
    print(json.dumps(result, indent=4))
    
except requests.exceptions.HTTPError as err:
    print(f"\n❌ HTTP Error: {err}")
    print(f"Details: {response.text}")
except requests.exceptions.RequestException as err:
    print(f"\n❌ Connection Error: {err}")