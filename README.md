# Customer Churn Predictor

A web app that predicts whether a bank customer is likely to leave (churn) and explains *why*. A tuned Random Forest model is served through a **FastAPI** backend with an **HTML front end**, packaged with **Docker** and deployed on **Render**.

**Live demo:** (https://customer-churn-predictor-2s8r.onrender.com/)

> The app is on Render's free tier, so it may take up to a minute to wake up on the first visit after a period of inactivity.

## Features

- Web form for entering a customer's details and getting an instant churn prediction
- Churn probability shown with a gauge, plus a confidence score
- **SHAP explanations**: the top factors that pushed this customer's risk up or down
- JSON API with automatic interactive docs at `/docs`
- Input validation with Pydantic (e.g. geography, card type and satisfaction score are restricted to valid values)
- Dockerised for the same behaviour locally and in the cloud

## How it works

1. The user fills in the form in `index.html`.
2. The page sends the values as JSON to `POST /predict`.
3. The FastAPI app runs them through the saved scikit-learn pipeline (preprocessing + Random Forest) to get a prediction and probability.
4. A SHAP `TreeExplainer` calculates how much each feature contributed to the churn probability for that customer.
5. The page shows the verdict, the probability and the six biggest drivers. Bars to the right raise churn risk and bars to the left lower it.

## Dataset

Bank customer churn records (`data/Customer-Churn-Records.csv`), sourced from Kaggle: (https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn).

Each row is a customer, with the target `Exited` (renamed to `Churned`) showing whether they left the bank.

| Group | Features |
|---|---|
| Customer | `Age`, `Gender`, `Geography`, `CreditScore`, `EstimatedSalary` |
| Account | `Balance`, `Tenure`, `NumOfProducts`, `CardType`, `PointEarned`, `HasCrCard` |
| Engagement | `IsActiveMember`, `SatisfactionScore` |

**Data preparation**
- Dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- Dropped `Complain`: it tracks churn almost one-to-one, which looks like **data leakage**, so keeping it would inflate the results without helping real predictions
- Numeric features are median-imputed and standardised; categorical features are mode-imputed and one-hot encoded, all inside a single scikit-learn `Pipeline`

## Modelling

All of this is in [`model/EDA.ipynb`](model/EDA.ipynb).

**Data split:** stratified 90% train/validation and 10% test; the train/validation part is split again so the validation set is about 10% of the total.

**Baselines** (validation set): Logistic Regression, Random Forest, XGBoost.

**Tuning:** the Random Forest was tuned with [Optuna](https://optuna.org/) (50 trials, TPE sampler, 5-fold stratified cross-validation), searching over `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` and `class_weight`. It was tuned twice, once optimising **recall** and once optimising **F1**. The F1-optimised model was chosen as the final model because churn is the minority class and F1 balances catching churners against false alarms.

**Final model:** the tuned Random Forest was refit on the full training data (train + validation) and evaluated once on the untouched test set, then saved as `model/best_model_f1.pkl`.

### Results

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|
| Logistic Regression (validation) | 0.8121 | 0.61 | 0.22 | 0.33 |
| Random Forest (validation) | 0.8717 | 0.84 | 0.46 | 0.59 |
| XGBoost (validation) | 0.8545 | 0.72 | 0.47 | 0.57 |
| Tuned Random Forest, F1 (validation) | 0.8374 | 0.60 | 0.63 | 0.61 |
| **Tuned Random Forest, F1 (test)** | 0.8210 | 0.55 | 0.67 | 0.60 |


## Project structure

```
customer-churn-predictor/
├── app/
│   ├── app.py                   # FastAPI app (prediction + SHAP)
│   └── index.html               # Front end
├── data/
│   └── Customer-Churn-Records.csv
├── model/
│   ├── EDA.ipynb                # EDA, modelling and tuning
│   └── best_model_f1.pkl        # Saved trained pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

## Deployment

The app is deployed on [Render](https://customer-churn-predictor-2s8r.onrender.com/) as a web service connected to this GitHub repo, so every push to `main` can trigger a redeploy.

Render settings (fill in whichever you used):

| Setting | Value |
|---|---|
| Runtime | _Docker (uses the `Dockerfile`) or Python_ |
| Build command (Python runtime) | `pip install -r requirements.txt` |
| Start command (Python runtime) | `uvicorn app.app:app --host 0.0.0.0 --port $PORT` |

Render provides the `PORT` environment variable, so the app should listen on it rather than a hard-coded port.

## Tech stack

Python, FastAPI, Pydantic, scikit-learn, XGBoost, Optuna, SHAP, pandas, joblib, HTML/CSS/JavaScript, Docker, Render.

## Limitations and next steps

- The model is trained on a single bank's historical data, so it may not generalise to other customer populations
- `Complain` was excluded to avoid leakage, which means predictions can't use complaint information
- Try threshold tuning to trade precision against recall for retention campaigns
- Add automated tests for the `/predict` endpoint
- Add tests and CI that run before each Render deploy
