from fastapi import FastAPI, Query, Path, Body, Cookie, Header, HTTPException
from pydantic import BaseModel, Field, field_validator, BeforeValidator, AfterValidator, ConfigDict
from typing import Annotated, Union, List, Any
import datetime
import uuid
import re
import decimal
import uvicorn  # Import uvicorn

app = FastAPI()

# Pydantic Models
class DateInput(BaseModel):
    date_str: datetime.date

    @field_validator("date_str", mode="before")
    def convert_date_str_to_datetime(cls, value):
        try:
            print(type(value))
            # return value
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        except Exception as e:
            print("got+error" , e)
            raise ValueError("Invalid date format. Expected YYYY-MM-DD")

class ItemQueryModel(BaseModel):
    item_query: str = Field(alias="item-query", description="Search query for items")
    min_price: float = Field(default=0.0, ge=0)
    max_price: float = Field(default=100.0, le=100)

class Product(BaseModel):
    product_id: uuid.UUID
    name: str
    price: decimal.Decimal
    description: str | None = None
    created_at: datetime.datetime

class User(BaseModel):
    user_id: uuid.UUID
    username: str
    preferences: dict[str, Any]

class OrderItem(BaseModel):
    product: Product
    quantity: int

class Order(BaseModel):
    order_id: uuid.UUID
    items: List[OrderItem]

class DeepNestedItem(BaseModel):
    name: str
    details: dict[str, Any]

class DeepNestedOrder(BaseModel):
    order_id: uuid.UUID
    items: List[List[DeepNestedItem]]

class UserPreferences(BaseModel):
    theme: str = Field(examples=["dark", "light"])
    language: str = Field(examples=["en", "fr"])

class UserWithPreferences(BaseModel):
    user_id: uuid.UUID
    username: str
    preferences: UserPreferences

    model_config = ConfigDict(extra="forbid")

# Endpoints
@app.get("/date/")
async def get_date(date_input: DateInput):
    return {"date": date_input.date_str}

@app.get("/items/")
async def read_items(item_query: ItemQueryModel = Query(...)):
    return item_query

@app.get("/regex-test/")
async def regex_test(
    regex_param: Annotated[str, Query(regex=r"^[a-zA-Z0-9]+$")]
):
    return {"regex_param": regex_param}

@app.get("/deprecated-api/", deprecated=True)
async def deprecated_api():
    return {"message": "This API is deprecated"}

@app.get("/validate-data/")
async def validate_data(data: str):
    @AfterValidator
    def additional_validation(v: str) -> str:
        if len(v) < 5:
            raise ValueError("Data must be at least 5 characters long")
        return v

    validated_data = additional_validation(data)
    return {"validated_data": validated_data}

@app.post("/body-query/")
async def body_query(query_params: ItemQueryModel = Body(embed=True)):
    return query_params

@app.post("/nested-order/")
async def create_order(order: Order):
    return order

@app.post("/deep-nested-order/")
async def create_deep_nested_order(order: DeepNestedOrder):
    return order

@app.post("/user-preferences/")
async def set_preferences(user: UserWithPreferences, session_id: str = Cookie(...), user_agent: str = Header(convert_underscores=False)):
    return {"user": user, "session_id": session_id, "user_agent": user_agent}

@app.get("/byte-test/")
async def byte_test(byte_data: bytes = Query(...)):
    return {"byte_data": byte_data.decode()}

@app.get("/decimal-test/")
async def decimal_test(decimal_data: decimal.Decimal = Query(...)):
    return {"decimal_data": decimal_data}

@app.get("/any-test/", response_model=Any)
async def any_test():
    return {"any_data": [1, "string", {"key": "value"}]}

@app.get("/exclude-unset/", response_model=User, response_model_exclude_unset=True)
async def exclude_unset():
    return User(user_id=uuid.uuid4(), username="testuser", preferences={})

@app.get("/")
async def root():
    return {"message": "FastAPI application started successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)