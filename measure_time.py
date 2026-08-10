import time
import os
from dotenv import load_dotenv
load_dotenv()

from graph import run_research

queries = [
    "Impact of social media on mental health",
    "Future of electric vehicles globally"
]

times = []
for i, query in enumerate(queries, 1):
    print(f"\n{'='*50}")
    print(f"Query {i}/{len(queries)}: {query}")
    print('='*50)

    start = time.time()
    result = run_research(query)
    elapsed = round(time.time() - start, 1)
    times.append(elapsed)

    print(f"\n✅ Time taken: {elapsed}s")

    if i < len(queries):
        print("Waiting 90 seconds before next query...")
        time.sleep(90)

print(f"\n{'='*50}")
print(f"PARALLEL RESULTS")
print(f"{'='*50}")
for i, (q, t) in enumerate(zip(queries, times), 1):
    print(f"Query {i}: {t}s  —  {q}")
print(f"Average: {round(sum(times)/len(times), 1)}s")