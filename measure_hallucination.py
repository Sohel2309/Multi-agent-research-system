import os
from dotenv import load_dotenv
load_dotenv()

from graph import run_research

queries = [
    "Latest developments in quantum computing 2025",
    "Impact of remote work on productivity",
    "Growth of electric vehicles in India"
]

for i, query in enumerate(queries, 1):
    print(f"\n{'='*60}")
    print(f"Query {i}/3: {query}")
    print('='*60)
    
    result = run_research(query)
    
    print("\n--- QA VERDICT ---")
    print(result['qa_review'])
    
    print("\n--- REPORT PREVIEW (first 800 chars) ---")
    print(result['report'][:800])
    
    if i < 3:
        input("\nPress Enter when ready for next query (wait 90s first)...")