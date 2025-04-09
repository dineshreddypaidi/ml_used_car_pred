import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import inference

infer = inference.Inference()
result = infer.predict(
    fuel_type="Petrol",
    seller_type="Individual",
    transmission_type="Manual",
    brand="Maruti",
    model="Ecosport",
    vechile_age = 2015,
    km_driven = 45000,
    mileage = 20,
    engine = 700,
    max_power = 70,
    seats = 5
)


print("Predicted price:", result)