# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    history_items = relationship("HistoryItem", back_populates="owner", cascade="all, delete-orphan")

class HistoryItem(Base):
    __tablename__ = "history_items"
    id = Column(Integer, primary_key=True, index=True)
    pergunta = Column(String, index=True)
    resumo_ia = Column(Text)
    resultados = Column(JSON) 
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="history_items")