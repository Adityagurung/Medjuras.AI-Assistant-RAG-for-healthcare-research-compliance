# data

```
data/
├── raw/           # HF downloads
│   └── medrag/
├── processed/     # records + chunks
│   ├── medrag/
│   └── chunks/
└── evaluation/    # ground_truth.json
```

Index volumes (`esdata/`, `qdrant_storage/`) are created by Docker under `data/` when you run `docker compose up`.
