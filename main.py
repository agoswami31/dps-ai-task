from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

app = FastAPI()

# Pydantic model for request body validation
class PredictionInput(BaseModel):
    year: int
    month: int

# Load the pre-trained model and scaler
model_path = "model_rf.pkl"   
scaler_path = "scaler_rf.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# API endpoint for for welcome message
@app.get("/")
def welcome():
    return {"message": "Welcome to DPS AI Challenge Server, to make predictions go to /predict route!"}

# API endpoint for prediction
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Check if the provided month is within the valid range (1 to 12)
        if 1 <= input_data.month <= 12:
            # Make predictions for the given input
            custom_input = np.array([[input_data.year, input_data.month]])
            custom_input_scaled = scaler.transform(custom_input)
            predicted_value = model.predict(custom_input_scaled)

            # Extract the prediction value
            prediction_value = predicted_value[0]

            # Create the response body
            response_body = {"prediction": prediction_value}
            return response_body
        else:
            raise HTTPException(status_code=422, detail="Invalid month. Please provide a month in the range 1 to 12.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Run the FastAPI application using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)