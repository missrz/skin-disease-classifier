from motor.motor_asyncio import AsyncIOMotorClient
import os
from typing import Optional

try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	# dotenv is optional; environment variables can be provided by the environment
	pass

# Environment-configurable values
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
# PROJECT_NAME can be used to derive a project-specific DB name if DATABASE_NAME not set
PROJECT_NAME = os.environ.get("PROJECT_NAME", "skin_disease_classifier")
DATABASE_NAME = os.environ.get("DATABASE_NAME", f"{PROJECT_NAME}_db")

client: Optional[AsyncIOMotorClient] = None
db = None

def init_db(uri: Optional[str] = None, db_name: Optional[str] = None):
	"""Initialize the Motor client and return the database instance.

	Calling this is optional; the module will lazily initialize on first import/use.
	"""
	global client, db
	if client is None:
		client = AsyncIOMotorClient(uri or MONGO_URI)
	db = client[db_name or DATABASE_NAME]
	return db

def get_db():
	"""Return the database instance, initializing the client if necessary."""
	global client, db
	if db is None:
		return init_db()
	return db

def close_client():
	"""Close the Motor client connection (call at shutdown)."""
	global client
	if client is not None:
		client.close()
