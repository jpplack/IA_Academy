# security.py
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext
from pathlib import Path
from .import schemas 

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY não encontrada no .env. (Rode 'openssl rand -hex 32' para gerar uma)")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):

    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception: Exception) -> Optional[schemas.TokenData]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    return token_data