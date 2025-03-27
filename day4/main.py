from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.routing import APIRouter
from sqlmodel import SQLModel, create_engine, Session, Field
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import time
import httpx
import secrets
import uvicorn
import os

# ✅ Security & Authentication Config
SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ✅ Database Setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# ✅ OAuth2 Setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", scopes={"admin": "Admin access", "user": "Regular user access"})
password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ✅ FastAPI Instance
app = FastAPI(
    openapi_url="/openapi.json",  # 🔹 Default OpenAPI URL
    docs_url="/docs",             # 🔹 Default Swagger UI
    redoc_url="/redoc"            # 🔹 Default ReDoc UI
)

# ✅ Middleware for Timer
@app.middleware("http")
async def timer_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    print(f"Request: {request.url} took {process_time:.4f}s")
    return response

# ✅ CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Database Model
class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    hashed_password: str
    role: str

# ✅ Create Database Tables Before App Starts
def create_db():
    SQLModel.metadata.create_all(engine)

create_db()  # Ensure database is initialized

# ✅ Dependency for DB Connection
def get_db():
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()

# ✅ Helper Functions
def hash_password(password: str):
    return password_context.hash(password)

def verify_password(plain_password, hashed_password):
    return password_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ✅ Token Authentication Route
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token({"sub": user.username}), "token_type": "bearer"}

# ✅ API Router (No Global Dependency)
router = APIRouter(prefix="/api")

# ✅ Protected Route (Requires Authentication)
@router.get("/users/me")
def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"message": "User authenticated", "token": token}

# ✅ Background Tasks Example
def send_email(email: str, message: str):
    print(f"Sending email to {email}: {message}")

@router.post("/send-email")
def schedule_email(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email, email, "Welcome!")
    return {"message": "Email scheduled"}

app.include_router(router)

# ✅ Test Endpoint
@app.get("/test")
def test_endpoint():
    with httpx.Client() as client:
        response = client.get("https://jsonplaceholder.typicode.com/posts/1")
        return response.json()

# ✅ Static Files
# if not os.path.exists("static"):
#     os.makedirs("static")

# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "FastAPI application started successfully!"}


# ✅ Run the App Correctly
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
