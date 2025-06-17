# Task 2 - Data Version Control Setup

This task focuses on establishing a reproducible and auditable data pipeline using Data Version Control (DVC).

## Setup

1. Install DVC:
```bash
pip install dvc
```

2. Initialize DVC:
```bash
dvc init
```

3. Set up local remote storage:
```bash
mkdir dvc-storage
dvc remote add -d localstorage dvc-storage
```

4. Add data:
```bash
dvc add data/insurance_data.csv
```

5. Commit changes:
```bash
git add data/*.dvc .dvc/config
git commit -m "Add data tracking with DVC"
```

## Project Structure

```
task-2/
├── data/              # Data directory (DVC tracked)
├── src/              # Source code
├── tests/            # Unit tests
├── .dvc/             # DVC configuration
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
``` 