# Workflow-CI

Workflow CI untuk *re-training* otomatis model **Telco Customer Churn**
menggunakan **MLflow Project** + **GitHub Actions** — submission MSML Kriteria 3.

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml      # GitHub Actions: re-train + artifact + Docker
└── MLProject/
    ├── MLProject                 # definisi MLflow Project (entry point)
    ├── conda.yaml                # environment (python 3.12.7 + deps)
    ├── modelling.py              # training script (autolog + manual metrics)
    ├── DockerHub.txt             # tautan image Docker Hub
    └── telco_preprocessing/      # dataset siap latih (train.csv, test.csv)
```

## Alur CI (Advanced)

1. **Trigger** — `push` ke `main` atau `workflow_dispatch`.
2. **Re-train** — `mlflow run MLProject --env-manager=local` melatih ulang model
   dan menyimpan artefak ke `mlruns/`.
3. **Simpan artefak** — model diunggah sebagai GitHub Actions artifact
   (`actions/upload-artifact`).
4. **Docker** — `mlflow models build-docker` membungkus model menjadi Docker
   image, lalu di-`push` ke Docker Hub (`masterA88/telco-churn-model`).

## Secrets yang dibutuhkan

| Secret | Keterangan |
|--------|------------|
| `DOCKER_HUB_USERNAME` | username Docker Hub |
| `DOCKER_HUB_TOKEN`    | access token Docker Hub |

## Menjalankan secara lokal

```bash
cd MLProject
mlflow run . --env-manager=local
```

---
*Author: Hilmi — https://master-hilmi.vercel.app/*
