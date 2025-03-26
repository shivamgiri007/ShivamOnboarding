from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi import BackgroundTasks
from typing import List, Optional
from datetime import datetime
import uvicorn

app = FastAPI()

@app.get("/status_code_example", status_code=status.HTTP_200_OK)
async def status_code_example():
    return {"message": "Success", "status_code": 200}

@app.post("/form_example")
async def form_example(name: str, age: int):
    return {"name": name, "age": age}

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "file_content": content[:50]}

@app.get("/raise_exception/{item_id}")
async def raise_exception(item_id: int):
    if item_id == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"message": "Validation Error", "detail": exc.errors()},
    )

@app.get("/response_example", tags=["Example Endpoints"], summary="Example Response", description="This is an example response endpoint.")
async def response_example():
    return {"message": "This is an example of status codes, responses, and description in FastAPI."}

class Item(BaseModel):
    name: str
    price: float
    timestamp: datetime

@app.get("/jsonable_encoder_example", response_model=Item)
async def jsonable_encoder_example():
    item = Item(name="Sample Item", price=123.45, timestamp=datetime.now())
    item_dict = jsonable_encoder(item)
    return item_dict

@app.put("/put_example/{item_id}")
async def put_example(item_id: int, item: Item):
    return {"item_id": item_id, "item": item}

@app.patch("/patch_example/{item_id}")
async def patch_example(item_id: int, item: Item):
    return {"item_id": item_id, "item": item}

@app.get("/dict_vs_model_dump/{item_id}")
async def dict_vs_model_dump(item_id: int):
    item = Item(name="Sample Item", price=99.99, timestamp=datetime.now())
    item_dict = item.model_dump()
    return {"item_id": item_id, "item_dict": item_dict}

def get_query_param(q: str = None):
    return {"query_param": q}

@app.get("/dependency_injection_example")
async def dependency_injection_example(query_params: dict = Depends(get_query_param)):
    return query_params

def sub_dependency(q: Optional[str] = None):
    return {"sub_param": q}

def common_dependency(sub_q: Optional[str] = Depends(sub_dependency)):
    return {"common_sub_param": sub_q}

@app.get("/sub_dependency_injection")
async def sub_dependency_injection(dep: dict = Depends(common_dependency)):
    return dep

@app.get("/dependencies_in_path_operation", dependencies=[Depends(common_dependency)])
async def dependencies_in_path_operation():
    return {"message": "This endpoint uses dependency injection in path operation decorator."}

@app.post("/send_email/")
async def send_email(background_tasks: BackgroundTasks, email: str):
    background_tasks.add_task(send_email_task, email)
    return {"message": "Email sending task has been started in the background."}

async def send_email_task(email: str):
    await some_email_sending_function(email)

async def some_email_sending_function(email: str):
    print(f"Sending email to {email}")

@app.get("/")
async def root():
    return {"message": "FastAPI application started successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
