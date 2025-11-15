# crud.py
from sqlalchemy.orm import Session
from .import models, schemas, security
from typing import List

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        username=user.username, 
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_history(db: Session, user_id: int) -> List[models.HistoryItem]:
    """
    Busca o histórico de um usuário específico, ordenado do mais novo para o mais antigo.
    """
    return (
        db.query(models.HistoryItem)
        .filter(models.HistoryItem.owner_id == user_id)
        .order_by(models.HistoryItem.timestamp.desc())
        .all()
    )

def create_user_history_item(db: Session, user_id: int, item: schemas.HistoryItemCreate) -> models.HistoryItem:
    db_item = models.HistoryItem(
        **item.model_dump(),
        owner_id=user_id
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def clear_user_history(db: Session, user_id: int) -> bool:

    try:
        num_rows_deleted = db.query(models.HistoryItem).filter(models.HistoryItem.owner_id == user_id).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Erro ao deletar histórico: {e}")
        return False