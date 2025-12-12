"""Utility functions for NICE NG203 guidelines ETL."""

# Imports
import json


# Load HTML file
def load_html(file_path):
    """Load HTML content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        html_str = str(f.read())
    return html_str

# Save JSON
def save_json(file_path, dict_list):
    """Save to JSON."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_list, f, indent=2)

# Load JSON
def load_json(file_path):
    """Load from JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        dict_list = json.load(f)
    return dict_list
