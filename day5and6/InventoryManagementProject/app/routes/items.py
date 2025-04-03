from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
import re
import os
import uuid
from sqlmodel import Session, select
from fastapi.responses import FileResponse

from app.db.models import Item, ItemCreate, ItemRead, User
from app.db.session import get_session
from app.core.security import get_current_active_user, get_current_user


IMAGES_DIR = "item_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

router = APIRouter(
    dependencies=[Depends(get_current_active_user)],
    responses={404: {"description": "Not found"}}
)

@router.post("/{item_id}/image")
async def upload_item_image(
    item_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user: str = Depends(get_current_user)
):
    # Verify item exists and user has permission
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Verify user owns the item
    user = session.exec(select(User).where(User.username == current_user)).first()
    if not user or item.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this item")
    
    # Generate unique filename
    file_ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(IMAGES_DIR, filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Update item with image path (you'd add an image_path field to your Item model)
    item.image_path = file_path
    session.add(item)
    session.commit()
    
    return {"filename": filename}

@router.get("/{item_id}/image")
async def get_item_image(item_id: int, session: Session = Depends(get_session)):
    item = session.get(Item, item_id)
    if not item or not item.image_path:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(item.image_path)

@router.post("/", response_model=ItemRead, status_code=status.HTTP_201_CREATED)
async def create_item(
    item: ItemCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user)
):
    db_item = Item(**item.model_dump()) 
    db_item.owner_id = current_user.id
    
    session.add(db_item)
    session.commit()
    session.refresh(db_item)
    return db_item

@router.put("/{item_id}", response_model=ItemRead)
async def update_item(
    item_id: int,
    item_update: ItemCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user)
):
    db_item = session.get(Item, item_id)
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    if db_item.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this item"
        )
    
    item_data = item_update.model_dump(exclude_unset=True)
    for key, value in item_data.items():
        setattr(db_item, key, value)
    
    session.add(db_item)
    session.commit()
    session.refresh(db_item)
    return db_item

