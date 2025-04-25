use upload_data.py to load the data in the milvus and posgressql 

after uploading file to milvus and postgres create venv 

install requiremetns.txt

run main.py 

ask question to model and type exit if want to exit.

SET .env File :
OPEAI_API_KEY="KEY HERE"
OPENAI_MODEL="gpt-4"
EMBEDDING_MODEL="text-embedding-3-large"

# Milvus (Docker)
MILVUS_HOST="localhost"
MILVUS_PORT="19530"
MILVUS_COLLECTION="knowledge_base"

# PostgreSQL (Local)

PG_HOST="localhost"
PG_PORT="5432"
PG_DATABASE="query_router"
PG_USER="postgres"
PG_PASSWORD="password"