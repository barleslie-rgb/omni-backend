from pathlib import Path
import shutil
from datetime import datetime
target=Path("main.py")
patch=Path("destination_discover_route.py")
marker='@app.get("/api/v1/destination-discover")'
if not target.exists(): raise SystemExit("Run this from C:\\omni_backend; main.py not found.")
if not patch.exists(): raise SystemExit("destination_discover_route.py must be in this folder.")
source=target.read_text(encoding="utf-8")
if marker in source: raise SystemExit("Route already exists; no changes made.")
for token in ("supabase","HTTPException","Query","import re"):
    if token not in source: raise SystemExit(f"Expected dependency not found ({token}); inspect main.py. No changes made.")
backup=target.with_name("main.py.before_destination_catalog_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".bak")
shutil.copy2(target,backup)
with target.open("a",encoding="utf-8",newline="\n") as f: f.write("\n\n"+patch.read_text(encoding="utf-8").strip()+"\n")
print("Added destination-discover route. Backup:",backup)
print("Run: python -m py_compile main.py && git diff --check; inspect diff before commit.")
