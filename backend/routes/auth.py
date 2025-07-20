from fastapi import APIRouter, HTTPException
from db.mongo import users_collection
from schemas.user import UserCreate, UserOut,UserLogin
from passlib.context import CryptContext
from bson import ObjectId
from utils.jwt import create_access_token


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()

@router.post("/register", response_model=UserOut)
async def register_user(user: UserCreate):
    # Check if email already exists
    existing = await users_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_password = pwd_context.hash(user.password)

    # Store in DB
    result = await users_collection.insert_one({
        "email": user.email,
        "password": hashed_password
    })

    return UserOut(id=str(result.inserted_id), email=user.email)


@router.post("/login")
async def login(user: UserLogin):
    existing_user = await users_collection.find_one({"email": user.email})
    if not existing_user:
        raise HTTPException(status_code=404, detail="Invalid email or password")
    
    if not pwd_context.verify(user.password, existing_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Generate JWT
    token = create_access_token({"user_id": str(existing_user["_id"])})
    
    return {"access_token": token, "token_type": "bearer"}