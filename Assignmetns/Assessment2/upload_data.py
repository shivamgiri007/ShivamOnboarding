from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
from pymilvus import MilvusClient, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class SPLADEEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        
    def embed(self, text: str) -> Dict[int, float]:
        """Generate SPLADE sparse embeddings"""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            output = self.model(**tokens)
        
        weights = torch.max(torch.log(1 + torch.relu(output.logits)), dim=1)[0].squeeze()
        cols = weights.nonzero().squeeze().cpu().numpy().astype(np.int64)
        weights = weights[cols].cpu().numpy().astype(np.float32)
        
        return dict(zip(cols.tolist(), weights.tolist()))

def process_pdf_to_milvus(pdf_path: str, collection_name: str = "knowledge_base"):
    """Process PDF file and store embeddings in Milvus using MilvusClient"""
    try:
        # Initialize Milvus client
        client = MilvusClient(
            uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}"
        )

        # Define collection schema
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1536)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="page", datatype=DataType.INT64)

        # Prepare index parameters
        index_params = client.prepare_index_params()

        # Add indexes
        index_params.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="AUTOINDEX",
            metric_type="L2"
        )

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2}
        )

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        # Load and split PDF
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        print(f"Split into {len(chunks)} chunks")
        
        # Initialize embedders
        dense_embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        sparse_embedder = SPLADEEmbedder()
        
        # Generate embeddings and prepare data
        print("Generating embeddings...")
        data = []
        for i, chunk in enumerate(chunks):
            # Generate embeddings
            dense_embedding = dense_embedder.embed_documents([chunk.page_content])[0]
            sparse_embedding = sparse_embedder.embed(chunk.page_content)
            
            # Prepare document
            doc = {
                "text": chunk.page_content,
                "dense": dense_embedding,
                "sparse": sparse_embedding,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", 0)
            }
            data.append(doc)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks")
        
        # Insert data in batches
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            client.insert(
                collection_name=collection_name,
                data=batch
            )
            print(f"Inserted batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
        
        print(f"Successfully inserted {len(data)} documents into collection '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False

def load_csv_to_postgres(csv_path: str, table_name: str):
    """Load CSV data into PostgreSQL"""
    try:
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Create engine
        engine = create_engine(
            f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@"
            f"{os.getenv('PG_HOST')}:{os.getenv('PG_PORT', '5432')}/"
            f"{os.getenv('PG_DATABASE')}"
        )
        
        # Load data
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Successfully loaded {len(df)} rows into table '{table_name}'")
        return True
        
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    pdf_success = process_pdf_to_milvus("IPL_Teams.pdf")
    # csv_success = load_csv_to_postgres("example.csv", "example_table")
    
    # if not pdf_success or not csv_success:
    #     print("Some operations failed - check error messages above")
    #     exit(1)
    
    print("All operations completed successfully")