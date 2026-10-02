import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYRAG_DATA_DIR", str(ROOT / "data_test"))
os.environ.setdefault("PYRAG_CHROMA_DIR", str(ROOT / "data_test" / "chroma"))
os.environ.setdefault("PYRAG_CACHE_DIR", str(ROOT / "data_test" / "cache"))
os.environ.setdefault("PYRAG_LLM_PROVIDER", "none")
os.environ.setdefault("PYRAG_ENABLE_WEB_SEARCH", "false")

from pyrag.llm import get_llm_or_fallback, resolve_provider  # noqa: E402

print("provider =", resolve_provider())
llm = get_llm_or_fallback()
print("llm =", type(llm))
print("invoke ->", str(getattr(llm.invoke("Contexto: teste"), "content", ""))[:80])

from pyrag.api import create_app  # noqa: E402

app = create_app()
print("--- routes ---")
for r in app.routes:
    print(getattr(r, "methods", ""), getattr(r, "path", r))

from fastapi.testclient import TestClient  # noqa: E402

with TestClient(app) as client:
    r1 = client.get("/.well-known/agent.json")
    print("card:", r1.status_code, str(r1.text)[:200])
    r2 = client.get("/a2a/health")
    print("health:", r2.status_code, str(r2.text)[:200])
    r3 = client.post("/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "tasks/list",
                                    "params": {}})
    print("rpc:", r3.status_code, str(r3.text)[:300])