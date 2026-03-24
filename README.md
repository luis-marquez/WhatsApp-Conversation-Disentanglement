# Temporal Clustering Engine for Privacy-Preserving WhatsApp Analytics

This repository contains the advanced Temporal Clustering Engine designed to partition streams of multi-agent communication into discrete conversation components based on variable temporal distributions.

The methodology implemented here focuses on robustness against sparse messaging patterns and integrates specialized mechanisms for managing isolates (single-message clusters) through adaptive resampling mapping.

## Repository Structure

To ensure maximal separation of concerns and adherence to complex software engineering standards, the engine has been refactored into a scalable modular architecture:

```
├── Dockerfile                  # Containerization specification for reproducible execution
├── README.md                   # This academic documentation
├── requirements.txt            # Explicit dependency locking
├── main.py                     # Entry point orchestrating the analytics pipeline
└── src/                        # Modular source code directory
    ├── clustering.py           # Core logic: Inter-quartile temporal algorithms and isolate resolution
    ├── config.py               # Centralized configuration dataclasses defining heuristics and paths
    ├── data_loader.py          # Data ingestion pipelines with support for .csv and .xlsx
    └── evaluation.py           # Mathematical evaluation library (ARI, NMI, V-Measure, Shen-F, etc.)
```

## Scientific Methodology

The clustering pipeline processes message sequences mapped by relative temporal intervals:
1. **Delta Generation**: Computes the first-order temporal derivative (time gap) between consecutive messages.
2. **Dynamic Thresholding**: Adapts an Inter-Quartile Range (IQR) threshold based on localized communication bursts. The heuristic identifies boundaries using a dynamically scaled `iqr * 29` boundary limit.
3. **Sender-bias Validation**: Incorporates identity continuity constraints—a gap exceeding a nominal 10-minute intra-sender threshold triggers partition boundary generation. Media items (e.g., images) have specialized reduced boundary thresholds (`irc/2`).
4. **Isolate Absorption**: A post-processing resolution pass reallocates isolated node clusters (size $n=1$) to the nearest temporal neighbor cluster via absolute minimal distance matching.

## Metrics & Evaluation

The module `src/evaluation.py` exposes a robust suite of validation metrics for evaluating unsupervised clustering algorithms against a known ground truth:
- **Entropy-based measures**: V-Measure, Setup-independent Normalized Mutual Info (NMI).
- **Pairwise measures**: Adjusted Rand Index (ARI), F1-Linkage.
- **Matching measures**: 1-to-1 Mapping ratio and Shen-F evaluation constraints.

## Docker Execution Instructions

For full reproducible results matching the conditions under which this research was conducted, utilizing Docker is highly recommended.

### 1. Build the engine

```bash
docker build -t temporal-clustering-engine .
```

### 2. Run the pipeline

Due to the hardcoded test paths in `src/config.py`, to run this correctly on an arbitrary machine, you should mount your data to the respective paths, or update `config.py` appropriately.

```bash
docker run --rm temporal-clustering-engine
```

*(Note: In a generalized deployment environment, update `src/config.py` or map volumes to your local directories as `-v /local/input:/app/input`)*

## Acknowledgements
This engine serves as the supplementary execution and analytical reference code accompanying the research article on data anonymization and conversation privacy preservation techniques.
