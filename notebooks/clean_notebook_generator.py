import re

with open("generate_notebook.py", "r", encoding="utf-8") as f:
    code = f.read()

# Regular expression to match '### 10. Interview Questions' up to but not including the closing triple quotes
clean_code, count = re.subn(r"### 10\. Interview Questions[\s\S]*?(?=\"\"\")", "", code)
print(f"Substituted {count} occurrences of Interview Questions.")

with open("generate_notebook.py", "w", encoding="utf-8") as f:
    f.write(clean_code)

print("Successfully saved cleaned generate_notebook.py")
