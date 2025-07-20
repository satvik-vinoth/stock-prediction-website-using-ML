from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env

MONGO_URL = os.getenv("MONGODB_URL")

client = AsyncIOMotorClient(MONGO_URL)
db = client["stockapp"]  # Choose your DB name

# Example collections you may use
users_collection = db["users"]
predictions_collection = db["predictions"]
stock_collection = db["stock_data"]