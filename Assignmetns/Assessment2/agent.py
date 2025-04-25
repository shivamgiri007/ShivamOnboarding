from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, List, Dict, Optional, Union, Tuple
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from pymilvus import MilvusClient
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import json
import logging
from typing import Literal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
_ = load_dotenv()

# Define the agent state
class AgentState(TypedDict):
    query: str
    query_type: str
    sql_results: Optional[List[Dict]]
    vector_results: Optional[List[Dict]]
    sql_confidence_score: Optional[float]  
    vector_confidence_score: Optional[float]
    best_source: Optional[str]
    response: Optional[str]
    rephrased_query: Optional[str]
    attempt_count: int
    max_attempts: int
    tables_metadata: Optional[str]
    reranked_results: Optional[List[Dict]]

# Initialize models and connections
model = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0,
    logprobs=True,  # Enable log probabilities for confidence scoring
    top_logprobs=5  # Get top 5 token probabilities
)

# Initialize Milvus client
milvus_client = MilvusClient(
    uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}"
)

# Initialize SQL database
engine_args = {
    "connect_args": {"connect_timeout": 5},
    "pool_pre_ping": True,  # Optional: helps with connection reliability
    "echo": False  # Set to True for debugging SQL queries
}

try:
    db = SQLDatabase.from_uri(
        f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT', '5432')}/{os.getenv('PG_DATABASE')}",
        engine_args=engine_args
    )
    print("Database connection established successfully")
except Exception as e:
    print(f"Failed to connect to database: {str(e)}")
    raise
# Initialize embedders
dense_embedder = OpenAIEmbeddings(model="text-embedding-3-small")

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

sparse_embedder = SPLADEEmbedder()

# Define prompts

SQL_QUERY_PROMPT = """You are an expert SQL query writer. Based on the following tables and their metadata:
{tables_metadata}

Write a SQL query to answer the following question. Only return the SQL query (WITHOUT any markdown formatting or backticks) with initial LIMIT=5. Change the LIMIT only if required.

Question: {query}

SQL Query:"""

SQL_CONFIDENCE_PROMPT = """Analyze the following SQL query results and determine how well they answer the user's question.
Consider completeness of response (are key fields missing?), data relevance (does it match the intent?), and result count.

User's question: {query}
SQL Query Results (first 5 rows shown): {sql_results}

Provide a confidence score between 0 and 1 where:
- 1 means perfect match, complete answer
- 0.7-0.9 means good match but might be missing some details
- 0.4-0.6 means partial match but needs more context
- 0.1-0.3 means very weak match
- 0 means no match at all

Also provide a brief explanation (1 sentence) of your confidence assessment.

Return your response in JSON format with keys 'confidence' and 'explanation'.
JSON Response:"""

VECTOR_CONFIDENCE_PROMPT = """Analyze the following Vector DB search results and determine how well they answer the user's question.
Consider semantic similarity, context coverage, and result quality.

User's question: {query}
Vector DB Results (first 3 shown): {vector_results}

Provide a confidence score between 0 and 1 where:
- 1 means perfect semantic match
- 0.7-0.9 means good semantic match
- 0.4-0.6 means partial match
- 0.1-0.3 means weak match
- 0 means no match

Also provide a brief explanation (1 sentence) of your confidence assessment.

Return your response in JSON format with keys 'confidence' and 'explanation'.
JSON Response:"""

REPHRASE_PROMPT = """The following query didn't return satisfactory results from our knowledge sources:
Original query: {query}

Please generate a semantically equivalent rephrased version that might work better. 
The rephrased version should maintain the same intent but use different wording or add context.

Rephrased query:"""

RESPONSE_GENERATION_PROMPT = """Based on the following information from our {source} knowledge base, answer the user's question.
If the information doesn't fully answer the question, say so and explain what's missing.

User's question: {query}
Confidence score: {confidence}/1
Explanation: {explanation}

Relevant information:
{results}

Your response:"""

# Define nodes


def preprocess_query(state: AgentState):
    """Node to preprocess and normalize the query"""
    query = state['query']
    
    # Basic preprocessing - in a real system we'd do more NLP here
    processed_query = ' '.join(query.lower().split())  # Simple normalization
    
    # If first attempt, get tables metadata
    if state['attempt_count'] == 0:
        tables_metadata = db.get_table_info()
        return {
            'query': processed_query,
            'tables_metadata': tables_metadata,
            'attempt_count': state['attempt_count'] + 1
        }
    
    return {'query': processed_query, 'attempt_count': state['attempt_count'] + 1}

def query_sql(state: AgentState):
    """Node to query SQL database and get results"""
    query = state['query']
    
    # System message for instructions
    system_msg = SystemMessage(content=SQL_QUERY_PROMPT.format(
        tables_metadata=state['tables_metadata'],
        query=""  # We'll put the actual query in the human message
    ))
    
    # Human message containing the actual query
    human_msg = HumanMessage(content=query)
    
    # Invoke model with both messages
    try:
        sql_query = model.invoke([system_msg, human_msg]).content
        # Clean up any potential markdown formatting
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        try:
            results = db.run(sql_query)
            # Try to parse as JSON, fallback to raw results if fails
            try:
                results_list = [dict(row) for row in json.loads(results)]
            except json.JSONDecodeError:
                # Handle case where results aren't in JSON format
                results_list = [{"result": results}]
            return {'sql_results': results_list}
        except Exception as e:
            print(f"SQL Execution Error: {e}")
            return {'sql_results': []}
    except Exception as e:
        print(f"SQL Generation Error: {e}")
        return {'sql_results': []}
    
def calculate_sql_confidence(state: AgentState):
    """Node to calculate confidence score for SQL results using log probabilities only."""
    if not state['sql_results']:
        return {'sql_confidence_score': 0.0}

    sample_results = state['sql_results'][:5]
    
    # You may include a light summary message, just to ground the LLM
    system_msg = SystemMessage(content="Evaluate how well these SQL query results match the user's question.")
    human_msg = HumanMessage(content=f"{state['query']}\n\nSQL Results Sample: {sample_results}")

    result = model.invoke([system_msg, human_msg])
    metadata = result.response_metadata

    try:
        confidence_score = calculate_weighted_confidence_score(metadata)
        return {
            'sql_confidence_score': confidence_score / 100  # Normalize to [0,1]
        }
    except Exception as e:
        return {
            'sql_confidence_score': 0.0,
            'sql_explanation': f"Logprob parsing failed: {str(e)}"
        }
    
def query_vector_db(state: AgentState):
    """Node to query Vector DB using hybrid search"""
    query = state['query']
    
    try:
        # Generate dense and sparse embeddings
        dense_embedding = dense_embedder.embed_query(query)
        sparse_embedding = sparse_embedder.embed(query)
        
        # Use the exact field names from your schema
        vector_field = "dense"  # Exact field name from schema
        sparse_field = "sparse"  # Exact field name from schema
        
        # Hybrid search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Prepare search request
        search_kwargs = {
            "collection_name": "knowledge_base",
            "data": [dense_embedding],
            "anns_field": vector_field,
            "search_params": search_params,
            "limit": 5,
            "output_fields": ["text", "source", "page"],
            "sparse_vector": sparse_embedding,
            "sparse_anns_field": sparse_field
        }
        
        # Execute search
        results = milvus_client.search(**search_kwargs)
        
        # Process results
        vector_results = []
        for hit in results[0]:
            vector_results.append({
                'text': hit.entity.get('text'),
                'source': hit.entity.get('source'),
                'page': hit.entity.get('page'),
                'score': hit.score
            })
        
        return {'vector_results': vector_results}
    
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}", exc_info=True)
        return {'vector_results': []}

def weighted_reranker(state: AgentState):
    """Node to rerank results using weighted combination of scores"""
    if not state['vector_results']:
        return {'reranked_results': state['sql_results']}
    if not state['sql_results']:
        return {'reranked_results': state['vector_results']}
    
    # Normalize scores (assuming SQL confidence is 0-1 and vector scores are similarity scores)
    sql_normalized = state['sql_confidence_score']
    vector_normalized = max(min(state['vector_confidence_score'], 1), 0)
    
    # Combine with weights (adjust these based on your needs)
    combined_confidence = 0.6 * sql_normalized + 0.4 * vector_normalized
    
    # Decide best source
    best_source = "SQL" if sql_normalized >= vector_normalized else "Vector"
    
    return {
        'best_source': best_source,
        'combined_confidence': combined_confidence
    }

def calculate_weighted_confidence_score(llm_results_metadata):
    """
    Calculate the weighted confidence score from the given LLM results metadata containing log probabilities.
    """
    log_probs = [item['logprob'] for item in llm_results_metadata['logprobs']['content']]
    probabilities = np.exp(log_probs)

    sorted_indices = np.argsort(log_probs)[-5:]

    joint_probability_all = np.prod(probabilities)
    top_probabilities = [probabilities[i] for i in sorted_indices]
    joint_probability_top_5 = np.prod(top_probabilities)

    confidence_score = round((0.7 * joint_probability_top_5 + 0.3 * joint_probability_all) * 100, 2)
    return confidence_score

def calculate_vector_confidence(state: AgentState):
    """Node to calculate confidence score for Vector DB results using log probabilities only."""
    if not state['vector_results']:
        return {'vector_confidence_score': 0.0}

    system_msg = SystemMessage(content="Analyze the vector results and generate a concise semantic summary.")
    human_msg = HumanMessage(content=state['query'])

    result = model.invoke([system_msg, human_msg])
    metadata = result.response_metadata

    try:
        # Calculate confidence using token log probabilities
        confidence_score = calculate_weighted_confidence_score(metadata)
        return {
            'vector_confidence_score': confidence_score / 100  # Normalize to [0,1]
        }
    except Exception as e:
        return {
            'vector_confidence_score': 0.0,
            'vector_explanation': f"Logprob parsing failed: {str(e)}"
        }

def generate_response(state: AgentState):
    """Node to generate final response based on best source"""
    if state['best_source'] == "SQL":
        results = state['sql_results']
        confidence = state['sql_confidence_score']
        explanation = state.get('sql_explanation', '')
    else:
        results = state['vector_results']
        confidence = state['vector_confidence_score']
        explanation = state.get('vector_explanation', '')
    
    system_msg = SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(
        source=state['best_source'],
        query="",  # Will be in human message
        confidence=confidence,
        explanation=explanation,
        results=results[:3]
    ))
    human_msg = HumanMessage(content=state['query'])
    
    response = model.invoke([system_msg, human_msg]).content
    
    return {'response': response}

def rephrase_query(state: AgentState):
    """Node to rephrase the query for another attempt"""
    system_msg = SystemMessage(content=REPHRASE_PROMPT.format(
        query=""  # Will be in human message
    ))
    human_msg = HumanMessage(content=state['query'])
    
    rephrased = model.invoke([system_msg, human_msg]).content
    
    return {'rephrased_query': rephrased, 'query': rephrased}

def should_retry(state: AgentState):
    """Conditional edge to determine if we should retry"""
    # If we have high confidence from either source, end
    if state.get('sql_confidence_score', 0) >= 0.7 or state.get('vector_confidence_score', 0) >= 0.7:
        return "end"
    
    # If we've reached max attempts, end
    if state['attempt_count'] >= state['max_attempts']:
        return "end"
    
    # Otherwise, retry with rephrased query
    return "rephrase"
def check_query_type(state: AgentState):
    """
    Use LLM to determine if the query is a DB/data-related query
    (i.e., should go through hybrid search) or just general.
    """
    prompt = f"""
You are an AI assistant that classifies user queries for a hybrid search system.
Decide if the query is related to database or information retrieval (including vector or SQL search),
or if it is just a casual or general question not needing database processing.

Query: "{state['query']}"

Respond with only one word: "structured" or "general"
"""

    llm_response = model.invoke(prompt).content.strip().lower()

    # Sanitize output to expected values
    query_type: Literal["structured", "general"] = (
        "structured" if "structured" in llm_response else "general"
    )

    return {
        "query": state["query"],
        "query_type": query_type
    }

def handle_general_prompt(state: AgentState):
    """Return response for general prompts."""
    return {
        "response": f"I'm here to help with database-related questions. You asked: '{state['query']}'"
    }

# Build the graph
builder = StateGraph(AgentState)

# Add nodes with unique names
builder.add_node("check_query_type", check_query_type)
builder.add_node("handle_general_prompt", handle_general_prompt)
builder.add_node("preprocess", preprocess_query)
builder.add_node("query_sql", query_sql)
builder.add_node("calculate_sql_conf", calculate_sql_confidence)
builder.add_node("query_vector", query_vector_db)
builder.add_node("calculate_vector_conf", calculate_vector_confidence)
builder.add_node("rerank", weighted_reranker)
builder.add_node("generate_response", generate_response)
builder.add_node("rephrase", rephrase_query)

# Set entry point
builder.set_entry_point("check_query_type")

builder.add_conditional_edges(
    "check_query_type",
    lambda state: state["query_type"],
    {
        "structured": "preprocess",
        "general": "handle_general_prompt"
    }
)

# Add edges
builder.add_edge("preprocess", "query_sql")
builder.add_edge("query_sql", "calculate_sql_conf")
builder.add_edge("calculate_sql_conf", "query_vector")
builder.add_edge("query_vector", "calculate_vector_conf")
builder.add_edge("calculate_vector_conf", "rerank")
builder.add_edge("rerank", "generate_response")

# Add conditional edge
builder.add_conditional_edges(
    "generate_response",
    should_retry,
    {
        "end": END,
        "rephrase": "rephrase"
    }
)

# Add edge from rephrase back to query_sql
builder.add_edge("rephrase", "query_sql")

# Compile the graph
graph = builder.compile()

# Example usage
if __name__ == "__main__":
    # Prepare inputs
    inputs = {
        'query': "Tell me about chennai super kings.",
        'max_attempts': 2,
        'attempt_count': 0,
        'tables_metadata': None
    }
    
    try:
        # Stream the execution
        for step in graph.stream(inputs):
            print("Current step output:")
            print(json.dumps(step, indent=2))
            print("---")
            
            # You can access specific state values like this:
            if 'response' in step:
                print(f"Current response: {step['response']}")
            if 'rephrased_query' in step:
                print(f"Rephrased query: {step['rephrased_query']}")
                
    except Exception as e:
        print(f"Error during graph execution: {str(e)}")
        