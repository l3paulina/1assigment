Assignment 1: GPS Spoofing Detection with Parallel Computing

## Goal

Analyze vessel tracking data from AIS records using **parallel computing** to detect GPS spoofing events. This project emphasizes:

- Efficient data handling and transformation  
- GPS spoofing detection via data analysis  
- Performance evaluation using Python’s parallel processing libraries

---

## Task Breakdown

### 1. Parallel Splitting of the Task
**Objective:** Break AIS data processing into parallelizable subtasks.  
**Guidance:**
- Explore different parallel computing strategies:  
  - **Data parallelism**
  - **Task parallelism**
- Ensure balanced workload across parallel tasks.

---

### 2. Implementation of Parallel Processing
**Objective:** Develop Python code that processes AIS data in parallel for speed and efficiency.  
**Guidance:**
- Use libraries like `multiprocessing`, `joblib`, or `concurrent.futures`
- Design code to allow configurable:
  - Number of CPUs
  - Chunk sizes

---

### 3. GPS Spoofing Detection

> GPS spoofing is the intentional manipulation of GPS signals, leading to incorrect location or time data — a critical issue for maritime safety.

#### A. Identifying Location Anomalies
- Detect **sudden and unrealistic jumps** in vessel location.

#### B. Analyzing Speed and Course Consistency
- Identify vessels with:
  - **Inconsistent speed patterns**
  - **Impossible distances** traveled in short intervals

---

### 4. Evaluating Parallel Processing Efficiency

**Objective:** Compare performance of sequential vs. parallel processing.

**Metrics to Analyze:**
- Execution time  
- CPU and memory usage  
- Speedup:  
  ```math
  \text{Speedup} = \frac{\text{Time (sequential)}}{\text{Time (parallel)}}

The repository has 3 parts:

1) Data extraction - how the data was downloaded and turned into a csv file.
2) Anomaly detection - main code for the parallel and sequential spoofing detection.
3) Parallel testing - testing the parallel spoofing detection process with different counts of cpu.
