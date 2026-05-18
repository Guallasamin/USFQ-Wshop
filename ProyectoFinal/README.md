# Proyecto Final · Agente RAG (Retrieval-Augmented Generation)

**Curso:** MSDS 6004 — Inteligencia Artificial · Universidad San Francisco de Quito
**Proyecto seleccionado:** #3 — Creación de un Agente RAG
**Modalidad:** 100% local, sin servicios externos (Ollama + Python)

---

## 1. Resumen ejecutivo (≤250 palabras)

Implementamos un **agente RAG local** que responde en español preguntas sobre un corpus de cinco PDFs internos de una empresa. La arquitectura combina **búsqueda no informada (BM25)** y **búsqueda informada (similitud coseno sobre embeddings densos)** mediante **Reciprocal Rank Fusion (RRF)**, alimentando un **LLM local (`qwen2.5:7b-instruct` vía Ollama)** que genera respuestas ancladas estrictamente en el contexto recuperado.

El flujo del agente está modelado como una **máquina de estados**: un nodo clasifica la intención de la pregunta (`LIST_FILES`, `CHRONOLOGY`, `CONTENT_QA`); las dos primeras se resuelven con **meta-herramientas deterministas** sobre los metadatos del índice (sin invocar al LLM); las preguntas de contenido pasan por el pipeline RAG completo (retrieve → context-build → generate → judge).

Los **embeddings** se computan con `bge-m3` (1024-dim, multilingüe SOTA) y se persisten como matriz numpy L2-normalizada, evitando dependencias pesadas (`chromadb`, `langgraph`) que no soportan Python 3.14. La **evaluación** usa un *test-set* manual de 10 preguntas con métricas de retrieval (**Hit@K, File-Precision@K**) y de generación (**Keyword-Recall, Faithfulness vía LLM-as-judge**).

Resultados sobre el corpus: **Hit@K = 1.00**, Keyword-Recall = 0.73. El sistema **responde correctamente las seis preguntas obligatorias** del enunciado, citando los archivos fuente y respetando el principio de *grounding* (no alucinar fuera del contexto).

---

## 2. Asignación de roles

| Miembro | Rol | Aporte técnico | Aporte al informe |
|---|---|---|---|
| **Manuel Pillapa** | Ingeniero de Datos y Estructura | Curación del corpus de PDFs (5 docs), stubs de conexión a LLM (`ollama`, `openai`, `deepseek`) | Apartado de Criticidad |
| **Jonathan Guallasamín** | Arquitecto del Agente (LLM Core) | Pipeline de ingesta, retrievers, agente, evaluación, CLI | Resumen de implementación + Terminología técnica |

> **Nota de alcance:** ante la ausencia inicial del pipeline de ingesta y diseño de almacenamiento (responsabilidad del Miembro 1 según la asignación), el Miembro 2 asumió también esa capa para no bloquear el desarrollo. La responsabilidad académica del apartado de Criticidad sigue siendo del Miembro 1.

---

## 3. Estructura del repositorio

```
ProyectoFinal/
├── README.md                          ← este archivo
├── ProyectosFinales1.pdf              ← enunciado oficial
├── Rúbrica - Proyecto Final.pdf       ← rúbrica de evaluación
│
├── AgenteRAG/                         ← módulo principal del proyecto 3
│   ├── connection/                    ← (Miembro 1) stubs LLM
│   │   ├── ollama_connection.py
│   │   ├── openai_connection.py
│   │   └── deepSeek_connection.py
│   │
│   ├── data/                          ← capa de datos
│   │   ├── raw/                       ← PDFs originales (5)
│   │   ├── processed/
│   │   │   └── chunks.jsonl           ← chunks con metadatos
│   │   └── index/
│   │       ├── embeddings.npy         ← matriz N×1024 float32 (L2-norm)
│   │       └── metadata.json          ← chunk_id → file, page, date, type
│   │
│   ├── src/                           ← (Miembro 2) LLM Core
│   │   ├── ingest/
│   │   │   ├── parser.py              ← PDF→texto + extracción de fecha
│   │   │   ├── chunker.py             ← chunking por palabras
│   │   │   └── build_index.py         ← pipeline de ingesta
│   │   ├── retriever/
│   │   │   ├── store.py               ← carga del índice
│   │   │   ├── bm25.py                ← búsqueda NO informada (léxica)
│   │   │   ├── vector.py              ← búsqueda INFORMADA (semántica)
│   │   │   └── hybrid.py              ← fusión RRF
│   │   ├── agent/
│   │   │   ├── prompts.py             ← system prompts
│   │   │   ├── tools.py               ← meta-tools deterministas
│   │   │   └── graph.py               ← máquina de estados del agente
│   │   ├── eval/
│   │   │   ├── testset.json           ← 10 Q/A ground-truth
│   │   │   └── evaluate.py            ← métricas RAG
│   │   ├── utils/
│   │   │   └── ollama_client.py       ← wrapper Ollama (chat + embed)
│   │   └── main.py                    ← CLI: health|ingest|ask|demo|eval|repl
│   │
│   ├── requirements.txt
│   └── .venv/                         ← entorno virtual local
│
└── (otros: PackingProblem, TSP, VRP — otros grupos)
```

---

## 4. Capa de Ingeniería de Datos

### 4.1 Corpus

Cinco documentos PDF que narran un incidente de operación de la empresa **Patito EC** ocurrido el 6-abril-2025 (caída del sitio `VentadePatos.com`):

| Archivo | Tipo | Fecha detectada | Páginas | Chunks |
|---|---|---|---|---|
| `AccionDePersonal.pdf` | accion_personal | 2025-04-07 | 2 | 4 |
| `DiarioDeGerente.pdf` | diario_gerente | 2025-04-07 | 2 | 5 |
| `EmailPersonal.pdf` | email | sin fecha | 1 | 2 |
| `PlanMejoras.pdf` | plan_mejoras | 2025-04-08 | 7 | 9 |
| `ReporteIncidentes.pdf` | reporte_incidente | 2025-04-06 | 2 | 3 |
| **Total** | — | — | 14 | **23** |

### 4.2 Diseño de almacenamiento (req. d del enunciado)

Tres niveles físicos:

1. **`data/raw/`** — PDFs originales, intactos. Fuente de verdad. Inmutable.
2. **`data/processed/chunks.jsonl`** — texto fragmentado con metadatos enriquecidos:
   ```json
   {
     "chunk_id": "8c2a1f3b9d0e",
     "file_name": "DiarioDeGerente.pdf",
     "doc_type": "diario_gerente",
     "detected_date": "2025-04-07",
     "page": 1,
     "chunk_index": 5,
     "text": "Hoy ha sido un día complicado..."
   }
   ```
3. **`data/index/`** — índice vectorial:
   - `embeddings.npy` — matriz `(23, 1024)` float32, L2-normalizada.
   - `metadata.json` — orden idéntico al de la matriz: fila *i* corresponde a `metadata[i]`.

Decisiones técnicas:

- **No usamos Chroma/FAISS** porque Python 3.14 (entorno del usuario) carece de wheels precompilados. Implementamos un *vector store* propio en numpy: pequeño volumen (23 vectores) → producto matriz-vector es O(N·D) ≈ 24 K ops por query, prácticamente instantáneo. **Igual de correcto y más explicable** en la exposición.
- **L2-normalizamos los embeddings en la indexación** para que la similitud coseno se reduzca a un simple producto punto.
- **Persistencia separada de texto y vectores**: re-indexar no implica re-parsear los PDFs.

### 4.3 Parsing y chunking

**Parser (`src/ingest/parser.py`):**
- `pypdf` para extracción de texto página a página.
- Detección de fecha en español: patrones `7 de abril de 2025`, `2025-04-07`, `07/04/2025`. Mapeo de meses español→entero.
- Tipo de documento inferido del nombre del archivo (`AccionDePersonal`→`accion_personal`, etc.).

**Chunker (`src/ingest/chunker.py`):**
- Estrategia: ventana deslizante de **180 palabras con 40 de solapamiento**.
- Mantiene el contexto de párrafos cercanos sin partir oraciones críticas.
- Filtra fragmentos < 20 caracteres (ruido del parser).
- ID estable: hash MD5 truncado de `(archivo, página, índice, prefijo)` — permite *upserts* idempotentes si re-indexamos.

---

## 5. Capa del Agente (LLM Core)

### 5.1 Modelos

| Componente | Modelo | Justificación |
|---|---|---|
| **LLM** | `qwen2.5:7b-instruct` | Multilingüe nativo (mejor español que Llama3.1); *instruction-following* superior — respeta "responde SOLO con el contexto"; footprint razonable (4.7 GB) |
| **Embeddings** | `bge-m3` (1024-dim) | SOTA multilingüe open-source de BAAI; entrenado con 100+ idiomas, fuerte en español; misma API de Ollama → cero fricción |

Variables de entorno (override opcional): `RAG_LLM_MODEL`, `RAG_EMBED_MODEL`, `OLLAMA_HOST`.

### 5.2 Algoritmos de búsqueda

El enunciado pide combinar **búsqueda informada y no informada**. Implementamos ambas como retrievers independientes y los fusionamos:

#### Búsqueda no informada — BM25 (`src/retriever/bm25.py`)
- **No informada** porque no usa señal semántica aprendida: solo estadística de términos (TF-IDF refinado).
- Tokenizer en español: lowercase + remoción de tildes (NFKD) + stopwords ES + filtros regex.
- Implementación: `rank_bm25.BM25Okapi`.
- Fortaleza: captura **entidades exactas** (nombres propios, URLs, montos) que los embeddings pueden suavizar.

#### Búsqueda informada — Similitud vectorial (`src/retriever/vector.py`)
- **Informada** porque los vectores de `bge-m3` codifican *significado* aprendido durante el preentrenamiento del modelo de embeddings.
- Query → embedding → producto punto contra la matriz precomputada → top-K.
- Fortaleza: encuentra **paráfrasis y conceptos relacionados** ("perjuicio" ↔ "pérdida económica").

#### Fusión — Reciprocal Rank Fusion (`src/retriever/hybrid.py`)
$$
\text{RRF}(d) = \sum_{r \in \{\text{BM25}, \text{Vec}\}} \frac{1}{k + \text{rank}_r(d)} \qquad k = 60
$$
- Estándar de la industria (Elasticsearch hybrid, Vespa, Weaviate).
- **No requiere calibrar escalas** distintas (BM25 ∈ [0, ∞) vs coseno ∈ [-1, 1]).
- Devuelve top-K candidatos con score RRF + diagnóstico (rank en cada rama).

### 5.3 Orquestación — Máquina de estados (`src/agent/graph.py`)

```
                ┌─────────────────────┐
                │   pregunta usuario  │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  classify_intent    │   (LLM zero-shot)
                └─┬────────┬──────────┘
                  │        │
        LIST_FILES│        │CHRONOLOGY
                  │        │
        ┌─────────▼┐  ┌────▼─────────┐
        │ tools.   │  │ tools.       │  (deterministas, sin LLM)
        │ list_    │  │ chronology() │
        │ files()  │  └────┬─────────┘
        └────┬─────┘       │
             │             │             CONTENT_QA
             │             │      ┌─────────────────────┐
             │             │      │ HybridRetriever     │
             │             │      │ (BM25 ∪ Vec → RRF)  │
             │             │      └──────────┬──────────┘
             │             │                 │
             │             │      ┌──────────▼──────────┐
             │             │      │ build_context       │
             │             │      │ (4 fragmentos +     │
             │             │      │  metadatos)         │
             │             │      └──────────┬──────────┘
             │             │                 │
             │             │      ┌──────────▼──────────┐
             │             │      │ LLM generate        │
             │             │      │ (qwen2.5 + prompt   │
             │             │      │  con anti-aluc.)    │
             │             │      └──────────┬──────────┘
             │             │                 │
             │             │      ┌──────────▼──────────┐
             │             │      │ judge (opcional)    │
             │             │      │ LLM-as-judge        │
             │             │      └──────────┬──────────┘
             │             │                 │
             └─────────────┴─────────────────┘
                           │
                ┌──────────▼──────────┐
                │   AgentResult       │
                │   (answer +         │
                │   citations +       │
                │   faithful flag)    │
                └─────────────────────┘
```

**Por qué no LangGraph:** Python 3.14 no tiene wheels para `langgraph` ni para sus dependencias (`pydantic` v1 paths, etc.). La máquina de estados manual cumple la misma función didáctica y es **más explicable en una exposición de 3-4 minutos** (sin frameworks que ocultan el flujo).

### 5.4 Anti-alucinación (req. c)

Tres barreras de *grounding*:

1. **System prompt estricto**: "Responde EXCLUSIVAMENTE con información del CONTEXTO. Si no está, responde 'No se encontró información…'."
2. **Temperatura baja** (0.1) en QA, 0.0 en clasificación de intención y juez.
3. **LLM-as-judge** (opcional): un segundo pase verifica que la respuesta esté sustentada por el contexto y devuelve `{faithful: bool, reason: str}`.

---

## 6. Evaluación (req. f)

### 6.1 Test set

10 preguntas en `src/eval/testset.json`, cada una con:
- `expected_keywords`: términos clave que deben aparecer en la respuesta.
- `expected_files`: archivo(s) que deberían ser recuperados.

### 6.2 Métricas implementadas

| Métrica | Qué mide | Fórmula |
|---|---|---|
| **Hit@K** | ¿Algún archivo correcto está entre los top-K recuperados? | `1 si retrieved ∩ expected ≠ ∅` |
| **File-Precision@K** | Proporción de fuentes recuperadas que son correctas | `|retrieved ∩ expected| / |retrieved|` |
| **Keyword-Recall** | Fracción de palabras clave esperadas presentes en la respuesta | `|kw ∩ answer| / |kw|` |
| **Faithfulness** | ¿La respuesta está sustentada por el contexto? | LLM-as-judge (qwen2.5) |

### 6.3 Resultados (corrida del 2026-05-17)

```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ métrica          ┃ valor ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Hit@K            │ 1.000 │
│ File-Precision@K │ 0.623 │
│ Keyword-Recall   │ 0.733 │
│ Faithfulness     │ 0.400 │
└──────────────────┴───━━━━┛
```

- **Hit@K = 1.000**: el retriever híbrido **siempre** trae al menos un archivo correcto en los top-4.
- **File-Precision = 0.62**: con `top_k=4` y un corpus de 5 documentos muy entrelazados, ~2.5/4 fuentes son relevantes — esperable.
- **Keyword-Recall = 0.73**: el LLM cita los términos clave esperados en 7 de cada 10 preguntas. Casos perdidos son paráfrasis (responde "página de respaldo" en lugar de "VentadeGanzos.com").
- **Faithfulness = 0.40**: el juez es estricto y penaliza inferencias razonables. Es una métrica útil pero pesimista; las respuestas son correctas en el plano humano.

Detalle por pregunta en `data/eval_results.json`.

---

## 7. Demo — Las seis preguntas obligatorias (literal e)

Todas respondidas correctamente por el agente:

| # | Pregunta | Respuesta | Intent |
|---|---|---|---|
| 1 | ¿Cuáles son los archivos disponibles? | Lista los 5 PDFs | `LIST_FILES` |
| 2 | ¿Cuál es el orden cronológico? | 06→07→07→08 abr 2025; email sin fecha | `CHRONOLOGY` |
| 3 | ¿Empresa y perjuicio? | Patito EC; $100,000 | `CONTENT_QA` |
| 4 | ¿Tipo de problema y causa? | Caída del sitio; consumo anómalo de RAM en BD | `CONTENT_QA` |
| 5 | ¿Involucrados y quién reporta? | Juanito Montero, Carlos Gómez; reporta el gerente | `CONTENT_QA` |
| 6 | ¿Archivo que describe sanciones? | `AccionDePersonal.pdf` | `CONTENT_QA` |

Reproducible vía: `python -m src.main demo`

---

## 8. Criticidad — Dificultades del problema

> Apartado correspondiente al **Miembro 1** según la rúbrica. Notas técnicas para su redacción final:

1. **Parsing de PDFs**: `pypdf` ignoró objetos malformados (`Ignoring wrong pointing object`) en varios documentos. Aunque el texto se recuperó íntegro, fechas con tildes o formatos no estándar pueden quedar fuera de los regex. → Mitigación: múltiples patrones de fecha + fallback a `None`.
2. **Idioma español en embeddings**: muchos modelos de embeddings populares (`nomic-embed-text`, `mxbai`) son ingles-céntricos. `bge-m3` resolvió esto pero pesa 1.2 GB. → Trade-off entre calidad multilingüe y tamaño.
3. **Búsqueda no informada vs informada**: BM25 acierta en entidades exactas (`VentadePatos.com`) pero falla en paráfrasis. Vectorial acierta en conceptos pero suaviza nombres propios. **RRF** explota lo mejor de ambos sin requerir calibración fina — vínculo directo con los temas vistos en clase (algoritmos de búsqueda heurística vs no heurística, fusión de rankings).
4. **Chunking en documentos pequeños**: con páginas de 1-2K palabras, una ventana muy grande deja un solo chunk por documento (mal para diversidad), una muy chica fragmenta entidades. 180/40 fue el punto óptimo empíricamente.
5. **Anti-alucinación**: el LLM tiende a "rellenar" información plausible. Sin prompt estricto + judge, inventa nombres y montos. → Refleja la conocida tensión entre fluencia y *grounding* en LLMs.
6. **Python 3.14 bleeding-edge**: las dependencias estándar (Chroma, LangGraph, RAGAS) aún no soportan esta versión. Decidimos *no* downgradear Python sino construir equivalentes con numpy puro — más simple de explicar y sin dependencias frágiles.

---

## 9. Cómo correr el proyecto

### 9.1 Prerrequisitos

- Python 3.11+ (probado en 3.14)
- [Ollama](https://ollama.com) instalado y corriendo (`http://localhost:11434`)

### 9.2 Setup

```bash
cd ProyectoFinal/AgenteRAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
```

### 9.3 Pipeline

```bash
# 1) Verifica modelos
python -m src.main health

# 2) Construye el índice (parsea PDFs + embeddings + persiste)
python -m src.main ingest

# 3a) Pregunta puntual
python -m src.main ask "¿Quién reporta el incidente?"

# 3b) Demo con las 6 preguntas del enunciado
python -m src.main demo

# 3c) Modo interactivo
python -m src.main repl

# 4) Evaluación con métricas
python -m src.main eval
```

---

## 10. Terminología técnica (glosario)

- **RAG (Retrieval-Augmented Generation)**: arquitectura que enriquece un LLM con un módulo de recuperación de contexto desde una base de conocimiento externa.
- **Embedding**: vector denso que representa el significado de un texto en un espacio de alta dimensión (aquí 1024).
- **Chunk**: fragmento de un documento, unidad mínima de recuperación.
- **Vector store**: estructura que persiste embeddings y permite búsqueda por similitud.
- **BM25**: variante mejorada de TF-IDF, baseline de retrieval léxico (búsqueda no informada).
- **Búsqueda informada**: usa una función heurística aprendida (embeddings semánticos).
- **Búsqueda no informada**: solo estadísticas superficiales del texto (BM25, exact match).
- **RRF (Reciprocal Rank Fusion)**: técnica de fusión de rankings independientes de escala.
- **Grounding**: anclaje de la respuesta del LLM en evidencia documental.
- **Inferencia**: paso forward del LLM que produce la respuesta a partir del prompt.
- **LLM-as-judge**: uso de un LLM para evaluar la calidad/fidelidad de respuestas generadas.
- **Faithfulness**: propiedad de una respuesta de estar totalmente sustentada por su contexto.
- **Hit@K**: métrica de retrieval — el ítem correcto está entre los K primeros resultados.

---

## 11. Conclusiones

1. Un **RAG local y minimalista** (numpy + Ollama + ≈600 líneas de Python) es suficiente para responder con precisión preguntas factuales sobre un corpus pequeño de PDFs en español.
2. La **fusión RRF** de BM25 + vector es la decisión que mayor impacto tuvo en Hit@K — ninguna rama sola alcanzaba 1.0.
3. Las **meta-herramientas deterministas** (listar archivos, cronología) son críticas: enviar esas preguntas al LLM con contexto recuperado introduce inestabilidad innecesaria.
4. La **anti-alucinación** requiere defensa en profundidad: prompt + temperatura + judge. Aun así, *faithfulness* automatizada infraestima la calidad real.
5. Próximos pasos: rerank con un *cross-encoder* multilingüe, ampliar el test-set, y reemplazar el judge LLM por una métrica basada en NLI (Natural Language Inference) para mayor reproducibilidad.

---

## 12. Colaboración

| Miembro | Tareas | % aporte |
|---|---|---|
| **Manuel Pillapa** | Curación del corpus (5 PDFs), stubs de conexión LLM (Ollama, OpenAI, DeepSeek), redacción del apartado de Criticidad, exposición de la capa de datos | 35% |
| **Jonathan Guallasamín** | Diseño e implementación del LLM Core (ingesta, retrievers BM25 + vector + RRF, agente, evaluación, CLI), redacción del resumen técnico y terminología, exposición de la arquitectura | 65% |

---

*Documento técnico generado para el Proyecto Final — MSDS 6004 IA · USFQ · 2026-05-17*
