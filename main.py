import os
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager
import secrets # For generating secure random tokens

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr # Import EmailStr for better email validation
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
# Replace with your MongoDB connection string if it's not local
MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017/")
DATABASE_NAME = "fastapi_auth_db"
USERS_COLLECTION_NAME = "users"
PASSWORD_RESET_TOKENS_COLLECTION_NAME = "password_reset_tokens" # Collection for reset tokens

# JWT configuration
# IMPORTANT: Generate a strong secret key for production (e.g., using 'openssl rand -hex 32')
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-replace-this-with-a-strong-one")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password Reset Token configuration
PASSWORD_RESET_TOKEN_EXPIRE_MINUTES = 60 # Password reset links expire in 60 minutes

# --- Database Connection Holder ---
# This class acts as a container for our MongoDB client and database objects.
# It allows us to initialize them once during startup and access them globally.
class Database:
    client: AsyncIOMotorClient = None
    db = None

db_client = Database()

# --- Lifespan Event Handler ---
# This function manages the application's lifecycle events (startup and shutdown).
# It's the recommended way in modern FastAPI to handle resource initialization and cleanup.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Connects to MongoDB on startup and closes the connection on shutdown.
    """
    # --- Startup Logic ---
    # Establish connection to MongoDB
    db_client.client = AsyncIOMotorClient(MONGO_DETAILS)
    # Select the specific database to work with
    db_client.db = db_client.client[DATABASE_NAME]
    print(f"Connected to MongoDB: {MONGO_DETAILS} database: {DATABASE_NAME}")
    
    # The 'yield' keyword passes control to the FastAPI application.
    # All code before 'yield' runs on startup.
    yield 
    
    # --- Shutdown Logic ---
    # All code after 'yield' runs on shutdown.
    # Close the MongoDB connection gracefully
    db_client.client.close()
    print("Disconnected from MongoDB.")

# --- FastAPI App Initialization ---
# The 'lifespan' context manager is passed to the FastAPI app,
# ensuring the startup and shutdown functions are automatically called.
app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
# Define allowed origins (your React app's URL)
origins = [
    "http://localhost:3000",  # React app's default address
    "http://127.0.0.1:3000",  # React app's default address (sometimes 127.0.0.1 is used)
    # Add your deployed frontend URL here when you deploy
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allow cookies/authorization headers to be sent
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allow all headers in the request
)

# --- Password Hashing ---
# CryptContext provides a secure way to hash and verify passwords using bcrypt.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain-text password against its stored bcrypt hash.
    Returns True if they match, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hashes a plain-text password using bcrypt.
    Automatically generates a unique salt for each hash.
    """
    return pwd_context.hash(password)

# --- JWT Token Functions ---
# OAuth2PasswordBearer helps extract the JWT token from the Authorization header.
# tokenUrl specifies where clients can obtain a token for Swagger UI.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a JWT (JSON Web Token) access token.
    The token contains the user's subject ('sub') and an expiration time ('exp').
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Default expiration if not specified
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    # Add the expiration timestamp to the token payload
    to_encode.update({"exp": expire})
    
    # Encode the payload into a JWT string using the secret key and algorithm
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Pydantic Models ---
# These models define the structure and validation rules for data.
# They are used for request bodies, response models, and internal data handling.

class UserBase(BaseModel):
    """Base model for user data, including common fields."""
    username: str = Field(..., min_length=3, max_length=50)
    # EmailStr provides built-in email format validation
    email: Optional[EmailStr] = None 

class UserCreate(UserBase):
    """Model for creating a new user (includes password for registration)."""
    password: str = Field(..., min_length=6)

class UserInDB(UserBase):
    """Model representing how user data is stored in the database."""
    # hashed_password is now required as social login is removed
    hashed_password: str 
    disabled: Optional[bool] = False

    class Config:
        # Allows Pydantic to handle MongoDB's ObjectId type without errors
        arbitrary_types_allowed = True 

class Token(BaseModel):
    """Model for the JWT token response after successful login."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Model for data extracted from a decoded JWT token."""
    username: Optional[str] = None

# --- Models for Password Reset ---
class PasswordResetRequest(BaseModel):
    """Model for requesting a password reset (user provides email)."""
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    """Model for confirming a password reset (user provides token and new password)."""
    token: str
    new_password: str = Field(..., min_length=6)

# --- User Authentication and Retrieval Functions ---
# These functions interact with the database to get user information.

async def get_user(username: str) -> Optional[UserInDB]:
    """
    Retrieves a user document from the 'users' collection by username.
    Returns a UserInDB object if found, None otherwise.
    """
    user_doc = await db_client.db[USERS_COLLECTION_NAME].find_one({"username": username})
    if user_doc:
        return UserInDB(**user_doc)
    return None

async def get_user_by_email(email: str) -> Optional[UserInDB]:
    """
    Retrieves a user document from the 'users' collection by email.
    Returns a UserInDB object if found, None otherwise.
    """
    user_doc = await db_client.db[USERS_COLLECTION_NAME].find_one({"email": email})
    if user_doc:
        return UserInDB(**user_doc)
    return None

async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Authenticates a user by checking their username and password.
    Returns the UserInDB object on success, False on failure.
    """
    user = await get_user(username)
    if not user:
        return False # User not found
    
    # Hashed password is always expected for local users
    if not verify_password(password, user.hashed_password):
        return False # Incorrect password
    
    return user # Authentication successful

# --- FastAPI Dependencies for Authentication ---
# These functions are used as dependencies in API routes to protect them.

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
    FastAPI Dependency: Decodes and validates a JWT token to get the current user.
    Raises HTTPException 401 if credentials are invalid or token is expired/malformed.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the JWT token using the secret key and algorithm
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub") # Extract the subject (username) from the payload
        
        if username is None:
            raise credentials_exception # Token missing essential 'sub' claim
        
        token_data = TokenData(username=username)
    except JWTError:
        # Catch any JWT-related errors (e.g., invalid signature, expired token)
        raise credentials_exception
    
    # Retrieve the full user details from the database using the username from the token
    user = await get_user(token_data.username)
    if user is None:
        # User not found in DB (e.g., user deleted after token was issued)
        raise credentials_exception
    
    return user # Return the authenticated user object

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """
    FastAPI Dependency: Ensures the current authenticated user is active (not disabled).
    Raises HTTPException 400 if the user is disabled.
    """
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

# --- API Endpoints ---

@app.post("/signup", response_model=UserBase, summary="Register a new user")
async def signup(user: UserCreate):
    """
    Registers a new user with a unique username and hashed password.
    Checks for existing username and email before creating the user.
    """
    # Check if username already exists
    existing_user = await db_client.db[USERS_COLLECTION_NAME].find_one({"username": user.username})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, # 409 Conflict indicates resource already exists
            detail="Username already registered"
        )
    
    # Check if email already exists (if provided)
    if user.email:
        existing_email_user = await db_client.db[USERS_COLLECTION_NAME].find_one({"email": user.email})
        if existing_email_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )

    # Hash the plain-text password before storing
    hashed_password = get_password_hash(user.password)
    
    # Create a UserInDB object for database storage
    user_in_db = UserInDB(username=user.username, email=user.email, hashed_password=hashed_password)

    # Insert the new user document into the MongoDB 'users' collection
    await db_client.db[USERS_COLLECTION_NAME].insert_one(user_in_db.dict(by_alias=True, exclude_none=True))
    
    # Return the UserBase representation (without hashed password)
    return user

@app.post("/token", response_model=Token, summary="Get an access token for login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticates a user using username and password (form data).
    Returns a JWT access token upon successful login.
    """
    # Authenticate the user using the provided credentials
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        # Raise 401 Unauthorized if authentication fails
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}, # Suggests Bearer token authentication
        )
    
    # Calculate token expiration time
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Create the JWT access token
    access_token = create_access_token(
        data={"sub": user.username}, # 'sub' claim identifies the user
        expires_delta=access_token_expires
    )
    
    # Return the access token and token type
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=UserBase, summary="Get current authenticated user's details")
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    """
    Retrieves the details of the currently authenticated and active user.
    This is a protected endpoint, requiring a valid JWT.
    """
    return current_user

@app.post("/logout", summary="Log out the current user")
async def logout(current_user: UserInDB = Depends(get_current_active_user)):
    """
    Logs out the current user. For JWTs, this primarily means the client discards the token.
    """
    return {"message": f"User {current_user.username} logged out successfully. Please discard your token."}

@app.post("/forgot-password", summary="Request a password reset link")
async def forgot_password(request: PasswordResetRequest):
    """
    Handles a request to reset a password.
    Generates a unique, time-limited token and simulates sending a reset link to the user's email.
    """
    user = await get_user_by_email(request.email)
    
    # Security best practice: Always return a generic success message
    # to prevent attackers from enumerating valid email addresses.
    if not user:
        print(f"Password reset requested for {request.email} (user not found or email not registered).")
        return {"message": "If a user with that email exists, a password reset link has been sent."}

    # Generate a cryptographically secure, URL-safe token
    reset_token = secrets.token_urlsafe(32)
    # Set token expiration time
    expires_at = datetime.utcnow() + timedelta(minutes=PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)

    # Store the token details in the 'password_reset_tokens' collection
    # 'user_id' links the token to the user (using username for simplicity)
    await db_client.db[PASSWORD_RESET_TOKENS_COLLECTION_NAME].insert_one({
        "user_id": user.username, 
        "token": reset_token,
        "expires_at": expires_at,
        "used": False # Flag to prevent token reuse
    })

    # --- Simulate sending email ---
    # In a real application, you would integrate with an email sending service (e.g., SendGrid, Mailgun).
    # The 'reset_link' would point to your frontend application's password reset page.
    reset_link = f"http://localhost:8000/reset-password?token={reset_token}" # Example link, replace with frontend URL
    print(f"\n--- SIMULATED EMAIL ---")
    print(f"To: {user.email}")
    print(f"Subject: Password Reset Request")
    print(f"Body: Click the following link to reset your password: {reset_link}")
    print(f"This link will expire in {PASSWORD_RESET_TOKEN_EXPIRE_MINUTES} minutes.")
    print(f"-----------------------\n")

    return {"message": "If a user with that email exists, a password reset link has been sent."}

@app.post("/reset-password", summary="Reset password using a token")
async def reset_password(request: PasswordResetConfirm):
    """
    Resets the user's password using a valid reset token.
    Validates the token's existence, expiration, and usage status.
    """
    # Find the token document in the 'password_reset_tokens' collection
    reset_token_doc = await db_client.db[PASSWORD_RESET_TOKENS_COLLECTION_NAME].find_one({
        "token": request.token
    })

    # If token not found, it's invalid or already deleted/expired
    if not reset_token_doc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token.")

    # Check if the token has expired
    if reset_token_doc["expires_at"] < datetime.utcnow():
        # Optionally, delete expired tokens to keep the collection clean
        await db_client.db[PASSWORD_RESET_TOKENS_COLLECTION_NAME].delete_one({"token": request.token})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token.")

    # Check if the token has already been used
    if reset_token_doc["used"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token already used.")

    # Get the user associated with this token
    user = await get_user(reset_token_doc["user_id"])
    if not user:
        # This scenario implies the user linked to the token was deleted.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User not found for this token.")

    # Hash the new password provided by the user
    hashed_new_password = get_password_hash(request.new_password)

    # Update the user's password in the 'users' collection
    # This updates the EXISTING user document, it does NOT create a new one.
    await db_client.db[USERS_COLLECTION_NAME].update_one(
        {"username": user.username}, # Filter: find the user document by username
        {"$set": {"hashed_password": hashed_new_password}} # Update: set the new hashed password
    )

    # Mark the token as used to prevent it from being reused
    await db_client.db[PASSWORD_RESET_TOKENS_COLLECTION_NAME].update_one(
        {"token": request.token},
        {"$set": {"used": True}}
    )

    return {"message": "Password has been successfully reset."}

@app.get("/", summary="Root endpoint")
async def read_root():
    """
    Basic root endpoint.
    """
    return {"message": "Welcome to the FastAPI Login System!"}

# --- How to Run ---
# To run this application, save the code as `main.py` and execute:
# uvicorn main:app --reload
#
# Then, you can access the API documentation at:
# http://127.0.0.1:8000/docs
