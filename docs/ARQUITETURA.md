# Arquitetura

```
┌─────────────┐
│  Documents  │  PDF/MD/HTML/CSV/URL
└──────┬──────┘
       │ ingest.py (load + chunk)
       ▼
┌──────────────────────────┐
│ HybridVectorStore        │
│  ├─ Chroma  (denso)      │
│  └─ BM25    (lexical)   │  → rerank heurístico → top-k
└──────┬───────────────────┘
       │ vectorstore.py
       ▼
┌──────────────────────────┐        ┌──────────────────────┐
│ RAGPipeline              │◄──────►│  Tools (agenticas)  │
│  query rewrite           │        │  rag_search          │
│  context + citations[n]  │        │  calculator          │
│  memória por sessão      │        │  web_search/fetch_url│
└──────────┬───────────────┘        │  sql_query           │
           │                        │  read_file           │
           │                        │  a2a_delegate        │
           │                        │  finish              │
           ▼                        └──────────┬───────────┘
   ┌───────────────┐                            │
   │ LLM gratuito  │                    ┌───────▼────────┐
   │ Ollama/Groq/  │                    │ ReactAgent     │
   │ Gemini/OpenR. │                    │ PENSE→ACTION   │
   └───────────────┘                    └───────┬────────┘
                                               │ A2A
                                    ┌──────────▼───────────┐
                                    │ A2AServer (JSON-RPC) │
                                    │ Agent Card           │
                                    └──────────┬───────────┘
                                               │ HTTP/SSE
                                    ┌──────────▼───────────┐
                                    │ A2AClient (peers)    │
                                    └──────────────────────┘
```

## Decisões de projeto

- **Híbrido (denso + lexical)**: BM25 recupera termos exatos (código, IDs, nomes) que
  embeddings oftenesmuram; oembeddings trazem semântica. Score final: `α·denso + (1-α)·lexical`
  seguido de rerank com cobertura de termos, densidade numérica e bônus de título.
- **Zero vendor lock**: embeddings e LLM são plugáveis (`llm.py`); trocar de provedor
  é só mudar `.env`.
- **Fallback extrativo**: sem nenhum LLM, o sistema ainda responde com os trechos
  recuperados — útil em CI, ambientes offline e testes.
- **A2A sobre JSON-RPC puro**: mesma base do Google A2A, sem SDK owner, para manter
  zero dependências extras e permitir falar com agentes de outros frameworks.
- **Segurança**: SQL somente-leitura validado por parsing; calculadora por AST (sem `eval`).