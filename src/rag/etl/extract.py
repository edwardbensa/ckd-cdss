"""Extraction functions for NICE NG203 guidelines."""

# Imports
import os
import re
import requests
from loguru import logger
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from src.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from src.rag.etl.utils import load_html, save_json


REC_URL = "https://www.nice.org.uk/guidance/ng203/chapter/recommendations"
RAT_URL = "https://www.nice.org.uk/guidance/ng203/chapter/rationale-and-impact"
GUIDELINES_HTML_DIR = EXTERNAL_DATA_DIR / "nice_guidelines"
os.makedirs(GUIDELINES_HTML_DIR, exist_ok=True)

rec_html_path = GUIDELINES_HTML_DIR / "nice_ng203_recommendations.html"
rat_html_path = GUIDELINES_HTML_DIR / "nice_ng203_rationale-and-impact.html"

GUIDELINES_DIR = PROCESSED_DATA_DIR / "nice_guidelines"
os.makedirs(GUIDELINES_DIR, exist_ok=True)

rec_json_path = GUIDELINES_DIR / "nice_ng203_recommendations.json"
rat_json_path = GUIDELINES_DIR / "nice_ng203_rationale-and-impact.json"


# Scrape NICE NG203 recommendations and rationale pages and save
def scrape_nice_ng203(output_path):
    """Scrape the main content from the NICE NG203 recommendations and rationale pages."""
    for chapter in ["recommendations", "rationale-and-impact"]:
        url = f"https://www.nice.org.uk/guidance/ng203/chapter/{chapter}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            logger.error(f"Failed to retrieve page: {response.status_code}")

        soup = BeautifulSoup(response.content, 'html.parser')
        content_div = soup.find('div', id='track-chapter') or soup.find('div', class_='chapter')

        file_path = output_path / f"nice_ng203_{chapter}.html"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(content_div))


# Parse recommendations from HTML
def parse_recommendations(html: str):
    """
    For each heading with class recommendation__number,
    get the recommendation body and the preceding section titles.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for tag in soup.find_all(["h4", "h5", "h6"], class_=re.compile(r'recommendation__number')):
        # Section
        section_tag = tag.find_previous("h3")
        section_str = section_tag.get_text(strip=True) if section_tag else None
        section_id = section_tag.get('id') if section_tag else None

        # Subsection
        subsection_tag = tag.find_previous("h4")
        subsection_str = (
            subsection_tag.get_text(strip=True)
            if subsection_tag and tag.name != "h4"
            else None
        )

        # Recommendation
        rec_id = tag.get_text(strip=True)
        body_tag = tag.find_next_sibling('div', class_='recommendation__body')
        if body_tag:
            rec_body = body_tag.get_text(" ", strip=True)
            rec_body = md(str(body_tag), heading_style="ATX", bullets="-")
            rec_body = rec_body.replace("\u00a0", " ")
            entry = {
                "guideline_id": "NG203",
                "rec_id": rec_id,
                "section": section_str,
                "subsection": subsection_str,
                "text": rec_body,
                "full_context": f"{section_str} > {subsection_str}: {rec_body}",
                "source_url": f"{REC_URL}#{section_id}" if section_id else REC_URL,
                "type": "recommendation"
            }
            results.append(entry)
    return results

# Parse tables from HTML
def parse_tables(html: str):
    """
    For each div with class informaltable,
    get the table content, caption, notes, preceding section titles.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for tag in soup.find_all("div", class_="informaltable"):
        # Section
        section_tag = tag.find_previous("h3")
        section_str = section_tag.get_text(strip=True) if section_tag else None
        section_id = section_tag.get('id') if section_tag else None

        # Subsection
        subsection_tag = tag.find_previous("h4")
        subsection_str = (
            subsection_tag.get_text(strip=True)
            if subsection_tag and subsection_tag != section_tag
            else None
        )

        # Table title (caption)
        title_tag = tag.find_next("caption")
        title_str = title_tag.get_text(strip=True) if title_tag else None

        # Table notes
        notes_tag = tag.find_previous("p")
        notes_str = notes_tag.get_text(strip=True) if notes_tag else None
        notes_str = notes_str if notes_str.startswith("Note:") else None # type: ignore

        # Table and combined text
        table_md = md(str(tag), heading_style="ATX", bullets="-")
        table_md = table_md.replace("\u00a0", " ")
        full_text = (
            f"### {title_str}\n\n **Notes:** {notes_str}\n\n{table_md}" if notes_str
            else f"### {title_str}\n\n{table_md}"
            )

        if not title_str:
            continue
        entry = {
            "guideline_id": "NG203",
            "table_id": f"table{len(results)+1}",
            "section": section_str,
            "subsection": subsection_str,
            "text": full_text,
            "full_context": f"{section_str} > {subsection_str}: {full_text}",
            "source_url": f"{REC_URL}#{section_id}" if section_id else REC_URL,
            "type": "table"
        }
        results.append(entry)
    return results

# Parse rationale and impact from HTML
def parse_rationales(html: str):
    """
    For each h3 heading with class title,
    get the children (h4, h5, p).
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for tag in soup.find_all("h3", class_="title"):
        # Section
        section_str = tag.get_text(strip=True)
        section_id = tag.get('id')

        # Rationale body
        body_tags = []
        for sibling in tag.find_next_siblings():
            if sibling.name == "h3":
                break
            body_tags.append(sibling)
        body_html = "".join(str(t) for t in body_tags)
        body_md = md(body_html, heading_style="ATX", bullets="-")
        body_md = body_md.replace("\u00a0", " ")

        entry = {
            "guideline_id": "NG203",
            "rationale_id": f"rtnl_{len(results)+1}",
            "section": section_str,
            "text": body_md,
            "full_context": f"{section_str}: {body_md}",
            "source_url": f"{RAT_URL}#{section_id}" if section_id else RAT_URL,
            "type": "rationale"
        }
        results.append(entry)
    return results


# Execution
if __name__ == "__main__":
    scrape_nice_ng203(GUIDELINES_HTML_DIR)
    rec_html: str = load_html(rec_html_path)
    recommendations = parse_recommendations(rec_html) + parse_tables(rec_html)
    rat_html: str = load_html(rat_html_path)
    rationales = parse_rationales(rat_html)
    save_json(rec_json_path, recommendations)
    save_json(rat_json_path, rationales)
