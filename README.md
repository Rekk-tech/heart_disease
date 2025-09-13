# Heart Disease Prediction Project

A comprehensive machine learning project for predicting heart disease risk using various algorithms and advanced techniques.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for heart disease prediction, including:

- **Data Analysis**: Comprehensive exploratory data analysis and visualization
- **Feature Engineering**: Advanced feature creation and selection techniques
- **Model Training**: Multiple algorithms including Naive Bayes, KNN, Decision Trees, and Ensemble methods
- **Model Evaluation**: Comprehensive evaluation with multiple metrics and visualizations
- **Model Explainability**: SHAP, LIME, and permutation importance analysis
- **Deployment**: FastAPI backend and Streamlit frontend for real-time predictions

## 📊 Dataset

The project uses the Cleveland Heart Disease Dataset with the following features:

- **Demographic**: Age, Sex
- **Medical History**: Chest pain type, resting blood pressure, cholesterol, fasting blood sugar
- **Test Results**: Resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope, major vessels, thalassemia
- **Target**: Heart disease presence (0: No disease, 1: Disease)

## 🏗️ Project Structure

```
heart_disease_project/
├── data/
│   ├── raw/                         # Heart_disease_cleveland_new.csv
│   ├── interim/                     # Intermediate processed data
│   └── processed/                   # Train/val/test splits, feature store
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA, statistics, class balance
│   ├── 02_visualization.ipynb       # Advanced visualizations
│   ├── 03_modeling.ipynb            # Compare NB, KNN, DT, Ensemble
│   └── 04_explainability.ipynb      # SHAP / permutation importance
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Load/validate data schema
│   ├── preprocessing.py             # Split + ColumnTransformer
│   ├── feature_engineering.py       # Feature creation and selection
│   ├── modeling.py                  # NB, KNN, DT, VotingClassifier
│   ├── evaluation.py                # ACC, F1, ROC-AUC, CM + plots
│   ├── persistence.py               # Save/load artifacts
│   ├── explain.py                   # SHAP/permutation/LIME
│   └── utils.py                     # Logger, seed, configuration
├── models/
│   ├── heart_model.pkl              # Trained ensemble model
│   ├── preprocessor.pkl             # ColumnTransformer
│   ├── quantiles.json               # Numeric feature percentiles
│   └── model_card.md                # Model description
├── app/
│   ├── api.py                       # FastAPI: /predict, /meta
│   └── ui_streamlit.py              # Streamlit form input & results
├── conf/
│   ├── config.yaml                  # Paths + numeric/categorical columns
│   └── params.yaml                  # Model hyperparameters
├── tests/
│   ├── test_preprocessing.py
│   ├── test_modeling.py
│   └── test_api.py
├── requirements.txt
├── README.md
├── Makefile                         # make train | api | ui | test
└── run_pipeline.py                  # Run full end-to-end pipeline
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd heart_disease_project

# Install dependencies
make setup
# or
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run the entire ML pipeline
make pipeline
# or
python run_pipeline.py
```

### 3. Start Applications

```bash
# Start API server (Terminal 1)
make api
# or
python app/api.py

# Start Streamlit UI (Terminal 2)
make ui
# or
streamlit run app/ui_streamlit.py
```

### 4. Access Applications

- **API Documentation**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501

## 📋 Available Commands

### Setup and Installation
```bash
make setup          # Install dependencies
make setup-dev      # Install development dependencies
```

### Data Operations
```bash
make data           # Load and validate data
make preprocess     # Preprocess data and create splits
```

### Model Training and Evaluation
```bash
make train          # Train models and create ensemble
make evaluate       # Evaluate models and generate reports
make explain        # Generate model explanations
```

### Applications
```bash
make api            # Start FastAPI server (localhost:8000)
make ui             # Start Streamlit UI (localhost:8501)
```

### Testing and Quality
```bash
make test           # Run all tests
make test-coverage  # Run tests with coverage report
make lint           # Run linting checks
make format         # Format code with black
```

### Jupyter Notebooks
```bash
make notebooks      # Start Jupyter server
make notebook-eda   # Open EDA notebook
make notebook-viz   # Open visualization notebook
make notebook-modeling # Open modeling notebook
make notebook-explain # Open explainability notebook
```

### Cleaning
```bash
make clean          # Clean temporary files
make clean-all      # Clean all generated files
```

## 🔧 Configuration

The project uses YAML configuration files:

- **`conf/config.yaml`**: Main configuration with paths, features, and settings
- **`conf/params.yaml`**: Hyperparameters for models and training

Key configuration sections:
- Data paths and feature definitions
- Model parameters and hyperparameters
- Evaluation metrics and cross-validation settings
- API and logging configuration

## 🧪 Testing

Run comprehensive tests:

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test modules
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_modeling.py -v
python -m pytest tests/test_api.py -v
```

## 📊 Model Performance

The ensemble model typically achieves:
- **Accuracy**: ~85-90%
- **F1 Score**: ~0.85-0.90
- **ROC-AUC**: ~0.90-0.95

## 🔍 Model Explainability

The project includes comprehensive model interpretability:

- **Feature Importance**: Tree-based feature importance
- **Permutation Importance**: Cross-validated feature importance
- **SHAP Values**: SHapley Additive exPlanations
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Partial Dependence**: Feature effect analysis

## 🌐 API Usage

### Predict Single Instance

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 50,
       "sex": 1,
       "cp": 0,
       "trestbps": 120,
       "chol": 200,
       "fbs": 0,
       "restecg": 0,
       "thalach": 150,
       "exang": 0,
       "oldpeak": 1.0,
       "slope": 1,
       "ca": 0,
       "thal": 0
     }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch-predict" \
     -H "Content-Type: application/json" \
     -d '[{...}, {...}]'
```

### Get Model Information

```bash
curl -X GET "http://localhost:8000/model-info"
```

## 🛠️ Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check
```

### Development Setup

```bash
# Setup development environment
make dev-setup

# Run CI pipeline
make ci
```

## 📈 Monitoring and Logging

- **Logs**: Stored in `logs/heart_disease.log`
- **Plots**: Saved in `plots/` directory
- **Model Artifacts**: Stored in `models/` directory

## ⚠️ Important Notes

### Medical Disclaimer

**This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.**

### Data Privacy

- The dataset contains anonymized medical data
- No personal identifiers are included
- All analysis is performed locally

### Model Limitations

- Trained on specific dataset and may not generalize to other populations
- Requires complete feature information for accurate predictions
- Clinical decisions should not be based solely on model predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Cleveland Heart Disease Dataset
- Scikit-learn library
- FastAPI and Streamlit frameworks
- SHAP and LIME libraries for model interpretability

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Made with ❤️ for healthcare and machine learning education**
