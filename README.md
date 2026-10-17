# Cacao Rating Prediction — MLOps Project

## Overview

This project develops a machine learning pipeline to predict chocolate bar ratings using the Flavors of Cacao dataset. It demonstrates an end-to-end MLOps workflow, including data validation, preprocessing, model training, experiment tracking, model registration, and continuous integration.

## Objectives

- Predict chocolate ratings using product and company-related features.
- Build a reproducible machine learning pipeline.
- Track experiments and evaluation metrics with MLflow.
- Validate input data and generated outputs.
- Automate pipeline execution using GitHub Actions.

## Dataset

**Dataset:** Flavors of Cacao

The dataset includes information such as:
- Company and chocolate bar name
- Review year
- Cocoa percentage
- Company location
- Bean type and origin
- Chocolate rating (prediction target)

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- MLflow
- Joblib
- Git and GitHub
- GitHub Actions
- Matplotlib

## Project Structure

```text
cacao-rating-mlops/
├── .github/
│   └── workflows/
│       └── lab8_ci.yml
├── data/
│   ├── flavors_of_cacao.csv
│   └── processed/
├── models/
├── outputs/
├── reports/
├── src/
│   ├── data_utils.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── preprocess_pipeline.py
│   ├── register_model.py
│   ├── train.py
│   ├── train_mlflow.py
│   ├── validate_data.py
│   └── validate_outputs.py
├── app.py
├── requirements.txt
├── run_lab3_baseline.py
├── run_lab4_tracking.py
├── run_lab5_pipeline.py
├── run_lab6_registry.py
└── run_lab7_pipeline.py
```

## MLOps Workflow

1. **Data validation:** Check the input dataset.
2. **Preprocessing:** Prepare features and target values for training.
3. **Model training:** Train regression models to predict chocolate ratings.
4. **Experiment tracking:** Use MLflow to record experiments and evaluation metrics.
5. **Model registration:** Register the trained model in the MLflow model registry.
6. **Output validation:** Check the generated pipeline outputs.
7. **Continuous integration:** Run the pipeline automatically through GitHub Actions.

## Evaluation Metrics

The regression models are evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of determination

These metrics measure prediction error and how well the model explains variation in ratings.

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Sasmitha1130/cacao-rating-mlop.git
cd cacao-rating-mlop
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the pipeline:

```bash
python run_lab7_pipeline.py
```

## Continuous Integration

The GitHub Actions workflow is defined in `.github/workflows/lab8_ci.yml`. It installs dependencies, runs the pipeline, and uploads available pipeline reports and outputs as workflow artifacts.

## Project Status

The project has been pushed to GitHub, and the GitHub Actions CI workflow has completed successfully.

## Future Improvements

- Deploy an interactive prediction application.
- Add automated model performance monitoring.
- Improve model performance through further experimentation.
- Add unit tests and stronger data-quality checks.

## Author

**Sasmitha Moilla**

Third-year undergraduate student — Artificial Intelligence and Machine Learning
