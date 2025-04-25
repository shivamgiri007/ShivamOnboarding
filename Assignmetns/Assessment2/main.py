from agent import graph
import logging
import json

logging.basicConfig(level=logging.INFO)
# Initialize the router


def main():
    print("Intelligent Query Router System")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        inputs = {
        'query': query,
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
   
    
        

if __name__ == "__main__":
    main()