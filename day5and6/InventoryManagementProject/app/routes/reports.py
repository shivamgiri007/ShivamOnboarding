from fastapi import APIRouter, Depends, BackgroundTasks
from datetime import datetime, timezone
from sqlmodel import Session
from typing import List
from app.db.models import Item
from app.db.session import get_session
from app.core.security import get_current_user
from app.tasks.notifications import send_email_notification, log_activity
import json

router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
    dependencies=[Depends(get_current_user)],
    responses={404: {"description": "Not found"}}
)

class InventoryReport(BaseModel):
    total_items: int
    total_value: float
    out_of_stock: int
    generated_at: datetime
    items: List[dict]

@router.post("/inventory", status_code=status.HTTP_202_ACCEPTED)
async def generate_inventory_report(
    background_tasks: BackgroundTasks,
    email: str = None,
    session: Session = Depends(get_session)
):
    # Generate report data
    items = session.exec(select(Item)).all()
    report_data = {
        "total_items": len(items),
        "total_value": sum(item.price * item.quantity for item in items),
        "out_of_stock": sum(1 for item in items if item.quantity == 0),
        "generated_at": datetime.now(timezone.utc),
        "items": [item.dict() for item in items]
    }
    
    # Add background tasks
    background_tasks.add_task(
        log_activity,
        f"Inventory report generated with {len(items)} items"
    )
    
    if email:
        background_tasks.add_task(
            send_email_notification,
            email,
            f"Inventory report generated at {report_data['generated_at']}"
        )
    
    return {"message": "Report generation started", "report_id": "12345"}