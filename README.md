# Hybrid ML-Digital Twin based Battery Recycling System

This project presents a hybrid AI-powered system designed to optimize material recovery in lithium-ion battery recycling. The system combines machine learning models, contamination detection, and digital twin simulation to predict chemical extraction, detect anomalies, and simulate batch behavior.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [File Structure](#file-structure)
- [Sample Output](#sample-output)
- [How to Run](#how-to-run)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

This CLI-based system models the chemical breakdown and reuse potential of battery black mass. It predicts recovery efficiency of lithium, nickel, and cobalt using Random Forest and Decision Tree regressors, and flags contaminated inputs using Isolation Forest. Additionally, it simulates batch moisture loss using Digital Twin principles and forecasts metal price trends to estimate recycling value.

---

## Features

- Predicts normalized chemical extraction outputs from industrial battery black mass.
- Simulates degradation, moisture loss, and reaction over time using digital twin logic.
- Detects contamination using Isolation Forest for quality assurance.
- Estimates final battery output potential (e.g., EV, mobile, laptop).
- Forecasts metal pricing trends using time-series data.
- Interactive CLI to guide batch predictions.

---

## Technologies Used

- Python, pandas, scikit-learn
- Random Forest, Decision Tree, Isolation Forest
- Digital Twin simulation logic
- SMOGN and ADASYN for dataset balancing
- Matplotlib for visualization
- CLI interface for batch processing

---

## File Structure

```
battery-recycling-ai/
├── data/
│   ├── Bag_dataset.csv
│   ├── Blackmass.csv
│   └── Metal_Prices_2021_2023.csv
├── Main.py
├── README.md
└── report/
    └── BRec_Final_Report.pdf
```

---

## Sample Output

```
Predicted Chemical Extraction:
  Lithium %: 7.62
  Nickel %: 9.87
  Cobalt %: 5.31

Contamination Detected: No
Simulated Moisture Loss: 1.27%
Estimated EV Batteries from batch: 21
Predicted Metal Price (Nickel): $21,400/tonne
```

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. **Run the script**
   ```bash
   python Main.py
   ```

3. **Follow CLI prompts** to load data and receive predictions.

---

## Documentation

A full technical report, including methodology, data sources, simulation design, and performance evaluation, is provided below:

Download: `./report/BRec_Final_Report.pdf`

---

## License

This project is provided under the [MIT License](LICENSE). You are free to use, modify, and distribute with proper attribution.
