#!/usr/bin/env bash
# Smoke test ponta a ponta: indexa, pergunta, ferramentas e A2A entre 2 agentes.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PASS=0
FAIL=0
ok()   { echo "  [OK]    $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FALHA] $1"; FAIL=$((FAIL+1)); }

echo "=== 1. Indexacao ==="
$PY -m pyrag index ./docs ./data/samples 2>&1 | tail -2
$PY -m pyrag status 2>&1 | head -12

echo
echo "=== 2. Busca hibrida (sem LLM) ==="
$PY -m pyrag rag search "BM25" --k 3 2>&1 | head -8

echo
echo "=== 3. Pergunta RAG ==="
$PY -m pyrag rag ask "o que e a comunicacao A2A?" 2>&1 | tail -14

echo
echo "=== 4. Memoria de sessao ==="
$PY -m pyrag rag ask "e o BM25?" --session smoke 2>&1 | tail -3
$PY -m pyrag rag ask "e de novo?" --session smoke 2>&1 | tail -3

echo
echo "=== 5. Ferramentas ==="
$PY - <<'PYEOF'
from pyrag.tools import calculator, list_documents, rag_search, sql_query
print("calc   :", calculator.invoke({"expression": "17*23"}))
print("calc2  :", calculator.invoke({"expression": "2*(3+4)-10/5"}))
print("sql x  :", sql_query.invoke({"sql": "DROP TABLE x"}))
print("rag    :", rag_search.invoke({"query": "A2A", "k": 1})[:70].replace(chr(10), " "))
print("files  :", list_documents.invoke({"path": "./docs"}).splitlines()[0])
PYEOF

echo
echo "=== 6. Agent Card ==="
$PY -m pyrag a2a card 2>&1 | head -5

echo
echo "=== 7. A2A entre 2 agentes (HTTP real) ==="
PYRAG_AGENT_NAME=agente-alpha PYRAG_A2A_BIND_PORT=9301 \
  $PY -m pyrag a2a serve >/tmp/a2a_alpha.log 2>&1 &
PID_A=$!
PYRAG_AGENT_NAME=agente-beta PYRAG_A2A_BIND_PORT=9302 \
  $PY -m pyrag a2a serve >/tmp/a2a_beta.log 2>&1 &
PID_B=$!
sleep 10

if curl -sf http://localhost:9301/.well-known/agent.json >/tmp/card_alpha.json; then
  ok "Agent Card do agente-alpha"
else
  bad "Agent Card do agente-alpha"
fi
if curl -sf http://localhost:9302/a2a/health >/dev/null; then
  ok "health do agente-beta"
else
  bad "health do agente-beta"
fi

if curl -sf -X POST http://localhost:9302/a2a -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{"role":"user","parts":[{"kind":"text","text":"ola"}]}}}' \
  > /tmp/rpc_send.json; then
  if grep -q '"state":"completed"' /tmp/rpc_send.json; then
    ok "message/send JSON-RPC 2.0"
  else
    bad "message/send payload inesperado: $(head -c 200 /tmp/rpc_send.json)"
  fi
else
  bad "message/send JSON-RPC 2.0"
fi

curl -sN -X POST http://localhost:9301/a2a -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"message/stream","params":{"message":{"role":"user","parts":[{"kind":"text","text":"oi"}]}}}' \
  > /tmp/rpc_stream.txt 2>&1
if grep -q 'data:' /tmp/rpc_stream.txt; then
  ok "message/stream SSE"
else
  bad "message/stream SSE"
fi
echo "  eventos SSE emitidos: $(grep -c 'data:' /tmp/rpc_stream.txt)"

TASK_ID=$($PY - <<'PYEOF' 2>/dev/null || true
import json
import urllib.request
data = json.load(urllib.request.urlopen("http://localhost:9302/a2a/tasks"))
print(data["tasks"][0]["id"] if data["tasks"] else "")
PYEOF
)
if [ -n "${TASK_ID:-}" ]; then
  if curl -sf -X POST http://localhost:9302/a2a -H 'Content-Type: application/json' \
      -d "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tasks/get\",\"params\":{\"id\":\"$TASK_ID\"}}" \
      | grep -q "$TASK_ID"; then
    ok "tasks/get"
  else
    bad "tasks/get"
  fi
else
  bad "tasks/get (nenhuma task encontrada)"
fi

echo "--- cliente A2A com descoberta pelo Agent Card ---"
if $PY - <<'PYEOF'
from pyrag.a2a.peers import discover_card

raw = discover_card("http://localhost:9302")
print("card keys:", sorted(raw.keys()))
print("skills no card:", [s["id"] for s in raw.get("skills", [])])
PYEOF
then
  ok "leitura do Agent Card remoto"
else
  bad "leitura do Agent Card remoto"
fi

if $PY - <<'PYEOF'
from pyrag.a2a.client import A2AClient
from pyrag.a2a.peers import add_peer, list_peers, resolve_peer

peer = add_peer("beta", "http://localhost:9302")
print("skills descobertas :", peer.skills)
print("resolve_peer(beta) :", resolve_peer("beta").url)
print("ping               :", A2AClient("http://localhost:9302").ping())
print("peers              :", [p.name for p in list_peers()])
PYEOF
then
  ok "registro de peer + ping"
else
  bad "registro de peer + ping"
fi

kill $PID_A $PID_B 2>/dev/null
wait 2>/dev/null

echo
echo "=============================="
echo " PASSOU: $PASS | FALHOU: $FAIL"
echo "=============================="
[ "$FAIL" -eq 0 ]