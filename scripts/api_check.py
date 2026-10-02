#!/usr/bin/env python3
"""Testa a API REST + A2A de um servidor pyrag em execucao (uso: api_check.py [base_url])."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
PASS = 0
FAIL = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  [OK]    {msg}")


def bad(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  [FALHA] {msg}")


def get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=20) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


def post(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:300]}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


print(f"=== API pyrag em {BASE} ===")

st, data = get("/health")
ok(f"/health -> {st} provider={data.get('provider')}") if st == 200 else bad(f"/health -> {st}")
if st == 200:
    idx = data.get("index", {})
    ok(f"indice: {idx.get('chunks')} chunks, {idx.get('bm25_docs')} docs BM25") \
        if idx.get("chunks") else bad("indice vazio")

st, tools = get("/tools")
names = [t["name"] for t in tools] if isinstance(tools, list) else []
ok(f"/tools -> {len(names)} ferramentas: {', '.join(names)}") if st == 200 else bad(f"/tools -> {st}")

st, card = get("/.well-known/agent.json")
if st == 200 and card.get("protocolVersion") == "0.3.0":
    ok(f"Agent Card: {card['name']} v{card['version']} "
       f"({len(card.get('skills', []))} skills)")
else:
    bad(f"Agent Card -> {st}")

st, _ = get("/a2a/health")
ok("/a2a/health") if st == 200 else bad(f"/a2a/health -> {st}")

st, res = post("/search", {"query": "A2A Agent Card", "k": 3})
if st == 200 and res.get("results"):
    r = res["results"][0]
    ok(f"/search -> {len(res['results'])} resultados; top score={r['score']:.3f} "
       f"em {r['source']['path'].split('/')[-1]}")
else:
    bad(f"/search -> {st} {res}")

st, res = post("/ask", {"question": "como funciona o rerank?", "session": "api-check"})
if st == 200 and res.get("answer"):
    ok(f"/ask -> resposta com {len(res.get('citations', []))} citacoes, "
       f"{res.get('elapsed_ms')}ms, model={res.get('model')}")
else:
    bad(f"/ask -> {st}")

st, res = post("/a2a", {"jsonrpc": "2.0", "id": 1, "method": "message/send",
                        "params": {"message": {"role": "user",
                                               "parts": [{"kind": "text", "text": "teste A2A"}]}}})
if st == 200 and res.get("result", {}).get("status", {}).get("state") == "completed":
    art = res["result"]["artifacts"][0]["parts"][0]["text"]
    ok(f"A2A message/send -> completed; artifact com {len(art)} chars")
else:
    bad(f"A2A message/send -> {st} {str(res)[:200]}")

st, res = post("/a2a", {"jsonrpc": "2.0", "id": 2, "method": "tasks/list", "params": {}})
n = len(res.get("result", {}).get("tasks", [])) if st == 200 else 0
ok(f"A2A tasks/list -> {n} tasks registradas") if st == 200 else bad(f"tasks/list -> {st}")

st, res = post("/a2a", {"jsonrpc": "2.0", "id": 3, "method": "inexistente"})
ok("A2A erro JSON-RPC -32601") if res.get("error", {}).get("code") == -32601 \
    else bad(f"erro A2A inesperado: {res}")

st, res = post("/a2a", {"jsonrpc": "1.0", "id": 4, "method": "tasks/list"})
ok("A2A erro JSON-RPC -32600 (jsonrpc invalido)") \
    if res.get("error", {}).get("code") == -32600 else bad(f"erro A2A: {res}")

st, sess = get("/sessions")
ok(f"/sessions -> {len(sess)} sessoes: {sess}") if isinstance(sess, list) \
    else bad("/sessions")

print(f"\n  PASSOU: {PASS} | FALHOU: {FAIL}")
sys.exit(0 if FAIL == 0 else 1)