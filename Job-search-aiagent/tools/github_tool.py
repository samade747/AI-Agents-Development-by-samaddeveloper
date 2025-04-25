## 4. `tools/github_tool.py`
```python
import requests

def fetch_github(link: str) -> str:
    """Fetch GitHub bio & top languages."""
    username = link.rstrip("/").split("/")[-1]
    # User info
    u = requests.get(f"https://api.github.com/users/{username}").json()
    bio = u.get("bio", "")
    # Repo languages
    repos = requests.get(f"https://api.github.com/users/{username}/repos").json()
    langs = {r.get("language") for r in repos if r.get("language")}
    return bio + "\nLanguages: " + ", ".join(sorted(langs))
```  