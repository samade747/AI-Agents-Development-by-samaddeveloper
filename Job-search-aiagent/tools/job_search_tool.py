## 6. `tools/job_search_tool.py`
```python
from serpapi import GoogleSearch
import os

def search_jobs(skills: list[str], location: str, job_type: str) -> list[dict]:
    """Query Google Jobs via SerpAPI and filter by type."""
    q = " ".join(skills)
    params = {
        "engine": "google_jobs",
        "q": q,
        "location": location or "",
        "job_type": job_type,  # "fulltime", "parttime", etc.
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    results = GoogleSearch(params).get_dict().get("jobs_results", [])
    return [j for j in results if job_type in j.get("job_type", "").lower()] or results
```  