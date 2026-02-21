# Text Classifier API (FastAPI + SQLite + Ollama embeddings)

Local-first text classification service.

- You define labels (name + definition).
- Incoming text is embedded and matched to the closest label by cosine similarity.
- Classified texts are stored as entries; label embeddings are continuously recomputed.

OpenAPI docs once running: `http://localhost:8000/docs`

## brief description of what is this project

This project is a small web app/API for **classifying text into labels**. You can add text entries, manage labels, and run a classification endpoint that assigns the most appropriate label(s) to new text. It’s useful for things like routing messages, tagging support tickets, organizing notes, or any workflow where you want consistent categories applied to free-form text.
I havent implemented frontend app for this project, but having api makes the integration easy.

## What’s in this repo

- **FastAPI** app in `app/main.py`
- **SQLite** database by default (`classifier.db`)
- **Ollama** used for embeddings (configurable)

## API routes

- `GET /health` → health check
- `POST /classify` → classify text (optional forced label)
- `GET /labels` → list labels
- `GET /labels/{name}` → label details + example entries
- `POST /labels` → create a label (computes embeddings)
- `DELETE /labels/{name}?force=true|false` → delete label (detaches entries first)
- `DELETE /entries/{entry_id}` → delete a stored entry (recomputes label embedding)
- `GET /stats` → counts (labels / classified entries / unclassified entries)

## Quickstart

From repo root:

1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Ensure `.env` exists

```dotenv
DATABASE_URL=sqlite:///./classifier.db
SIMILARITY_THRESHOLD=0.5
OLLAMA_HOST=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:8b-fp16
OLLAMA_TIMEOUT_SECONDS=20
```

3) Run the API

```bash
./scripts/run_dev.sh
```

## Classification behavior

### Matching mode (default AI matching)

`POST /classify` with only `text`:

- Embeds the input text.
- Computes similarity vs each label centroid.
- If best score $\ge$ `SIMILARITY_THRESHOLD` → stores the entry and returns `reason="matched_existing_label"`.
- Else → returns **422** with `best_match_label`/`best_match_score` and **does not store** an entry.

Safety rule: the service does not store “unlabelled” entries.

### Forced label mode

`POST /classify` with `label` or `label_id`:

- The label must already exist or the API returns **404**.
- Stores the entry with `reason="forced_label_assigned"`.

### Label centroids

Each label has a centroid embedding used for matching:

- If a label has no entries: centroid = normalized(definition embedding)
- If it has entries: centroid = normalized(0.5 × normalized(definition embedding) + 0.5 × normalized(entries centroid))

Centroids are recomputed after:

- successful classification into a label
- deleting an entry that belonged to a label
- creating a label

## Examples

Create a label:

```bash
curl -X POST "http://localhost:8000/labels" \
	-H "Content-Type: application/json" \
	-d '{"name":"Groceries and shopping","definition":"Buying food, groceries, and household shopping"}'
```

Classify (matching mode):

```bash
curl -X POST "http://localhost:8000/classify" \
	-H "Content-Type: application/json" \
	-d '{"text":"Buy bananas and grapes"}'
```

Force-assign by name (must already exist):

```bash
curl -X POST "http://localhost:8000/classify" \
	-H "Content-Type: application/json" \
	-d '{"text":"Buy bananas and grapes","label":"groceries_and_shopping"}'
```

Force-assign by id (must already exist):

```bash
curl -X POST "http://localhost:8000/classify" \
	-H "Content-Type: application/json" \
	-d '{"text":"Buy bananas and grapes","label_id":1}'
```

Delete an entry:

```bash
curl -X DELETE "http://localhost:8000/entries/123"
```

## Label naming rules

Label names are normalized to snake_case:

- lowercase
- whitespace → `_`
- non `[a-z0-9_]` characters → `_`
- collapsed `_` and trimmed

Example: `"Groceries and shopping"` becomes `"groceries_and_shopping"`.

## Ollama setup (required for embeddings)

This API uses Ollama for embeddings. Endpoints that require Ollama:

- `POST /classify`
- `POST /labels`
- `DELETE /entries/{entry_id}` (recomputes label centroid)

If Ollama is unavailable, these endpoints return:

- **503** when Ollama is unreachable
- **502** when Ollama returns an unexpected payload

Minimal setup:

1) Install and start Ollama: https://ollama.com
2) Pull an embedding model (must match your `.env`):

```bash
ollama pull qwen3-embedding:8b-fp16
```

3) Quick check:

```bash
ollama list
```

If you change `OLLAMA_EMBEDDING_MODEL` after you already have data in `classifier.db`, existing centroids may no longer be comparable (different embedding dimension/model behavior). In that case, either recompute centroids (not currently exposed as an endpoint) or recreate the DB.

## Database notes

- Default DB is SQLite: `classifier.db`.
- Tables are created on app startup.
- On startup, the app runs a small SQLite-only schema upgrader (`app/db/schema.py`).

To reset locally, stop the server and delete `classifier.db`.

## Development

Run tests:

```bash
pytest
```

## Additional notes

- Label matching uses AI embeding, that means the more informations AI will have about the label, more precise it will be. 
- The more advanced the embedding model will be (quantization, embeding Vector size, model size, quality of learning) the more reliable matchings will be.
- Especially with longer Vectors, having a decent amount of examples already assigned to the label is a good way to assure corectness.
- Force assigning wrong labels, and bad definition of the label will cause big problems with matching

