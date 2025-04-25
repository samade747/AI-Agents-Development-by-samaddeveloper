## 5. `tools/linkedin_tool.py`

import requests
from bs4 import BeautifulSoup

def fetch_linkedin(link: str) -> str:
    """Scrape public LinkedIn summary & skills."""
    resp = requests.get(link)
    soup = BeautifulSoup(resp.text, "html.parser")
    # Summary
    summary = soup.select_one(".pv-about__summary-text")
    text = summary.get_text(strip=True) if summary else ""
    # Skills
    skills = [el.get_text(strip=True) for el in soup.select(".pv-skill-category-entity__name-text")] or []
    return text + "\nSkills: " + ", ".join(skills)
