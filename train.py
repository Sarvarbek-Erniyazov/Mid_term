# src/train.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline

from sklearn.metrics import root_mean_squared_error, r2_score 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR 

from .utils import setup_logger 

logger = setup_logger('TRAINING_MODULE')


PROCESSED_PATH = 'data/processed/autos_processed.csv'
ENGINEERED_PATH = 'data/engineered/autos_engineered.csv'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'


MODELS = {
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.1, max_iter=10000, random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "LinearSVR": LinearSVR(max_iter=10000, random_state=42, dual=True)
}

TARGET_COL = 'kilometer'


def create_pipeline(X: pd.DataFrame):
    """Berilgan Dataframe ustunlariga mos ColumnTransformer va Pipeline yaratadi."""
    
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough', # Boshqa ustunlar o'zgarishsiz qoldiriladi
        n_jobs=-1
    )
    
    return preprocessor

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    """Modelni o'qitish va baholashni amalga oshiradi."""
    
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    return r2, rmse, training_time, pipeline


def train_and_compare(df: pd.DataFrame, data_name: str, results_list: list, best_model_info: dict):
    """Berilgan ma'lumotlar to'plami bo'yicha barcha modellarni o'qitadi va taqqoslaydi."""
    
    logger.info(f"\n--- {data_name} Data To'plamida O'qitish Boshlandi ---")
    
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    current_preprocessor = create_pipeline(X_train)
    
    for model_name, model_instance in MODELS.items():
        r2, rmse, t_time, pipeline = evaluate_model(
            model_name, model_instance, X_train, X_test, y_train, y_test, current_preprocessor
        )
        
        logger.info(f"  > {model_name} | R2: {r2:.4f}, RMSE: {rmse:.0f}, Vaqt: {t_time:.1f}s")
        
        results_list.append({
            'Model': model_name,
            'Data': data_name,
            'R2': r2,
            'RMSE': rmse,
            'TrainingTime': t_time
        })
        
        
        if r2 > best_model_info['R2']:
            best_model_info.update({
                'Model': model_name,
                'Data': data_name,
                'R2': r2,
                'RMSE': rmse,
                'Pipeline': pipeline
            })

    logger.info(f"--- {data_name} Data To'plamida O'qitish Yakunlandi ---")


def plot_results(results_df: pd.DataFrame):
    """Natijalarni vizuallashtiradi va saqlaydi."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    results_df_sorted = results_df.sort_values(by='R2', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    
    sns.barplot(x='Model', y='R2', hue='Data', data=results_df_sorted, ax=axes[0], palette='viridis')
    axes[0].set_title("Model Taqqoslash: R2 Balli (Eng Yuqori Yaxshiroq)")
    axes[0].set_xlabel('Regressiya Modeli')
    axes[0].set_ylabel('R2 Balli')
    axes[0].tick_params(axis='x', rotation=15)
    
   
    sns.barplot(x='Model', y='TrainingTime', hue='Data', data=results_df_sorted, ax=axes[1], palette='plasma')
    axes[1].set_title("Modelni O'qitishga Ketgan Vaqt (Soniya)")
    axes[1].set_xlabel('Regressiya Modeli')
    axes[1].set_ylabel('Vaqt (Soniya)')
    axes[1].tick_params(axis='x', rotation=15)
    
    axes[0].legend(title="Ma'lumot Turi", loc='upper left')
    axes[1].legend(title="Ma'lumot Turi", loc='upper left')
    plt.suptitle(f"Model Taqqoslash Tahlili (Jami {len(MODELS)} ta Model)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(REPORTS_DIR, 'model_comparison.png')
    plt.savefig(plot_path)
    logger.info(f"Natijalar grafigi '{plot_path}' ga saqlandi.")
    plt.close()


def run_model_comparison():
    """Asosiy taqqoslash jarayonini boshqarish."""
    
    if not os.path.exists(PROCESSED_PATH) or not os.path.exists(ENGINEERED_PATH):
        logger.error("XATO: Processed yoki Engineered data fayllari topilmadi. Avval 'run.py'ni ishga tushiring.")
        return None
    
    
    df_processed = pd.read_csv(PROCESSED_PATH)
    df_engineered = pd.read_csv(ENGINEERED_PATH)
    
    
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan).dropna()
    df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan).dropna()
    
    logger.info(f"Final Processed Data hajmi: {df_processed.shape}")
    logger.info(f"Final Engineered Data hajmi: {df_engineered.shape}")

    results_list = []
    best_model_info = {'R2': -np.inf, 'Pipeline': None} 
    
    train_and_compare(df_processed, 'Processed', results_list, best_model_info)
    
    
    train_and_compare(df_engineered, 'Engineered', results_list, best_model_info)

    
    results_df = pd.DataFrame(results_list)
    
   
    plot_results(results_df)

    
    if best_model_info['Pipeline']:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
        joblib.dump(best_model_info['Pipeline'], model_path)
        logger.info(f"Eng yaxshi Model Pipeline '{model_path}' ga saqlandi.")
        
        
        return {
            'Model': best_model_info['Model'],
            'Data': best_model_info['Data'],
            'R2': best_model_info['R2'],
            'RMSE': best_model_info['RMSE']
        }
    
    return None

if __name__ == '__main__':
    run_model_comparison()