# **Event-Related Potential (ERP) Analysis**

## **Project Overview**
This project analyzes **Event-Related Potentials (ERP)** from **ECoG (Electrocorticography) data**, focusing on **finger movement events**. The goal is to extract **ERP waveforms** from the ECoG signal and compute the **mean ERP response** for each finger.

## **Repository Structure**
```
📂 ERP_Analysis_Project
│── 📂 data
│   ├── brain_data_channel_one.csv        # ECoG brain signal data
│   ├── events_file_ordered.csv           # Event markers (start, peak, finger number)
│   ├── finger_data.csv                   # Processed data per finger
│
│── 📂 scripts
│   ├── erp_analysis.py                    # Main script to compute ERP
│   ├── plot_erp.py                        # Visualization script
│
│── 📂 results
│   ├── erp_mean_matrix.npy                # Saved ERP mean matrix
│   ├── erp_plot.png                        # Graph of ERP responses
│
│── README.md                              # Project documentation
```


## **Installation & Setup**
### **1. Clone Repository**
```bash
git clone <repository_url>
cd ERP_Analysis_Project
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
Make sure you have **Python 3.8+** installed.

### **3. Activate Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate    # Windows
```

## **Data Description**
The project uses two key CSV files:
- **brain_data_channel_one.csv**: Contains continuous **ECoG signals**.
- **events_file_ordered.csv**: Contains event markers (**start time, peak time, and finger number**).

**Goal**: Extract **ERP signals** around each movement event and compute the **mean ERP per finger.**


## **Usage**
### **1. Run ERP Analysis**
To compute ERP waveforms:
```bash
python scripts/erp_analysis.py
```
This script:
- Loads ECoG and event data
- Extracts trials for each finger
- Computes the **mean ERP waveform**
- Saves the result as `erp_mean_matrix.npy`

### **2. Visualize ERP Responses**
To plot the ERP waveforms:
```bash
python scripts/plot_erp.py
```
The ERP plot will be saved as **results/erp_plot.png**.


## **Main Code (erp_analysis.py)**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
events = pd.read_csv("data/events_file_ordered.csv")
brain_data = pd.read_csv("data/brain_data_channel_one.csv", header=None, names=["signal"])

# Extract event information
start_points = events.iloc[:, 0].values  # Start of movement
finger_labels = events.iloc[:, 2].values  # Finger number (1-5)

# Define time window
window_before, window_after = 200, 1000

# Store trials per finger
finger_trials = {finger: [] for finger in range(1, 6)}

# Extract trials
for i, start in enumerate(start_points):
    start = int(start)  # Ensure integer indexing
    if start >= window_before and start + window_after < len(brain_data):  
        trial_data = brain_data.iloc[start - window_before : start + window_after + 1].values.flatten()
        finger_trials[finger_labels[i]].append(trial_data)

# Compute mean ERP
fingers_erp_mean = np.array([np.mean(finger_trials[f], axis=0) for f in range(1, 6)])
np.save("results/erp_mean_matrix.npy", fingers_erp_mean)


## **Results**
- The **ERP mean matrix** is saved in `results/erp_mean_matrix.npy`.
- The **ERP plot** can be visualized using `plot_erp.py`.


## **Contributors**
- **Miya Estis**

## **Future Improvements**
- Support for **multiple ECoG channels**.
- Apply **signal filtering (e.g., band-pass filtering)**.
- Implement **statistical tests** to analyze ERP differences.

---

## **License**
This project is licensed under the MIT License. Feel free to use and modify it!

