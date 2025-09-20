# Automated OMR Evaluation & Scoring System

## Overview
This project automates the evaluation of OMR (Optical Mark Recognition) sheets using computer vision. It detects filled bubbles, calculates per-subject and total scores, and provides a web interface for fast, accurate evaluation. It reduces manual effort, errors, and turnaround time.

## Features
- Detects filled bubbles on OMR sheets captured via mobile or scanned images.
- Evaluates multiple sheet versions.
- Computes scores per subject and total marks.
- Web interface built with Streamlit for uploading OMR sheets and answer keys.
- Export results as CSV for record-keeping and audits.
- Handles ambiguous markings using simple thresholding and contour detection.

## Tech Stack
- **Python**: Core language for image processing and evaluation logic.
- **OpenCV**: Image preprocessing, contour detection, and bubble extraction.
- **NumPy / Pandas**: Array and data manipulation.
- **Streamlit**: Web interface for evaluators.
- **Pillow**: Image reading and conversion.
- **SQLite / CSV**: Storing and exporting results.

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
