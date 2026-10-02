#!/usr/bin/env bash
# Instala o projeto no venv e cria o .env
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Python: $(python3 --version)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> .env criado a partir de .env.example"
fi

mkdir -p data/chroma data/cache data/sessions

echo "==> Verificando Ollama..."
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "    Ollama rodando em http://localhost:11434"
  echo "    Modelos sugeridos: ollama pull llama3.1:8b qwen2.5:7b embeddinggemma:300m"
else
  echo "    Ollama NAO esta rodando. Inicie com: ollama serve &"
  echo "    (o sistema funciona em modo extrativo mesmo sem LLM)"
fi

echo
echo "Pronto. Ative o ambiente com: source .venv/bin/activate"
echo "Depois: pyrag index ./docs && pyrag rag ask 'o que este sistema faz?'"