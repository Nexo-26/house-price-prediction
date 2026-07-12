import json
import os

nb_path = "house_price_prediction.ipynb"
if os.path.exists(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    partial_cells = []
    for cell in nb["cells"]:
        source_text = "".join(cell["source"])
        if "## Phase 7" in source_text:
            break
        partial_cells.append(cell)

    nb["cells"] = partial_cells

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Truncated notebook successfully. Retained {len(partial_cells)} cells (Phases 1 to 6).")
else:
    print(f"Error: {nb_path} not found.")
