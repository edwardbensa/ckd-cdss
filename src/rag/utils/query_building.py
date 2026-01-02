import json

with open("src/rag/utils/tools.json", "r", encoding="utf-8") as f:
    tools = json.load(f)
print(tools)

