import os

from sqlalchemy import create_engine

# CONSTANTS
# postgresql connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

# Engine for connecting to postgresql database
ENGINE = create_engine(DATABASE_URL)

SCHEMA = 'qa'

qa_data_query = f"""
    SELECT * FROM {SCHEMA}.qa_data limit 100
"""