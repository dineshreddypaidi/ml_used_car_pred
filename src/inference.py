import joblib
import pandas as pd

def load_encoders(encoder_names: list, path_prefix: str):
    return {
        name: joblib.load(f"../models/encoders/{path_prefix}_{name}.pkl") for name in encoder_names
    }

class Inference():
    def __init__(self):
        self.encoders = load_encoders(["fuel_type", "seller_type","transmission_type","brand","model"], "encoder")
        self.model = joblib.load(f"../models/model.pkl")
                
    def preprocess(self, input_df):
        df = input_df.copy()
        for col in self.encoders:
            df[col] = self.encoders[col].transform(df[col])

        return df.values

    def predict(self, **kwargs):
        input_df = pd.DataFrame([kwargs])
        
        X = self.preprocess(input_df)
        
        prediction = self.model.predict(X)

        return round(prediction[0])
        