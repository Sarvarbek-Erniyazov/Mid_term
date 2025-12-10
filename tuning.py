# src/tuning.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time 
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score 


import optuna 
from optuna.samplers import TPESampler


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR 


try:
    
    import cuml.ensemble.RandomForestRegressor as cuRF
    import cuml.ensemble.GradientBoostingRegressor as cuGB
    print("cuML (GPU) kutubxonalari muvaffaqiyatli yuklandi. GPU rejimida ishlash imkoniyati mavjud.")
    GPU_AVAILABLE = True
except ImportError:
    # Sklearn (CPU versiyalari)
    print("cuML (GPU) topilmadi. CPU rejimida ishlanadi. Tezlik pasayishi mumkin.")
    cuRF = RandomForestRegressor
    cuGB = GradientBoostingRegressor
    GPU_AVAILABLE = False



try:
    from .utils import setup_logger 
    logger = setup_logger('TUNING_MODULE')
except ImportError:
    
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('TUNING_MODULE')



ENGINEERED_PATH = '../data/engineered/autos_engineered.csv' 
MODELS_DIR = '../models'
REPORTS_DIR = '../reports'
PARAMS_PATH = os.path.join(MODELS_DIR, 'best_params_tuned.json') 

TARGET_COL = 'kilometer'


ALL_MODELS = {
    "RandomForestRegressor": cuRF,
    "GradientBoostingRegressor": cuGB,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "KNeighborsRegressor": KNeighborsRegressor,
    "LinearSVR": LinearSVR,
}


def create_pipeline(X: pd.DataFrame):
    """Dataframe ustunlariga mos ColumnTransformer va Pipeline yaratadi."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough',
        n_jobs=-1
    )
    return preprocessor

def objective(trial, model_name, X_train, y_train, preprocessor):
    """Optuna ning maqsad funksiyasi (K-fold CV bo'yicha R2 ni oshirish)."""
    
    if model_name == "RandomForestRegressor":
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 8, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8]),
            'random_state': 42,
        }
        if not GPU_AVAILABLE:
            params['n_jobs'] = -1
        model = ALL_MODELS[model_name](**params)

    elif model_name == "GradientBoostingRegressor":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': 42,
        }
        model = ALL_MODELS[model_name](**params)

    elif model_name == "Ridge":
        params = {
            'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
            'random_state': 42
        }
        model = ALL_MODELS[model_name](**params)
        
    elif model_name == "Lasso":
        params = {
            'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'max_iter': 10000,
            'random_state': 42
        }
        model = ALL_MODELS[model_name](**params)
        
    elif model_name == "KNeighborsRegressor":
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2), 
            'n_jobs': -1
        }
        model = ALL_MODELS[model_name](**params)
        
    elif model_name == "LinearSVR":
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
            'loss': trial.suggest_categorical('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
            'max_iter': 10000,
            'random_state': 42,
            'dual': True 
        }
        model = ALL_MODELS[model_name](**params)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
    
    score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
    
    return score.mean()


def run_optuna_tuning(model_name, X_train, y_train, preprocessor, n_trials=50):
    """Optuna orqali giperparametr tuningini amalga oshiradi."""
    
    logger.info(f"  > Optuna tuningi boshlandi: {model_name} ({n_trials} urinish)")
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name=f'{model_name}_tuning')
    
    study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, preprocessor), 
                   n_trials=n_trials, 
                   show_progress_bar=False,
                   timeout=900) 
    
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"  > {model_name}: Eng yaxshi R2 (CV) balli: {best_score:.4f}")
    
    return best_params, best_score


def evaluate_final_model(model_name, best_params, X_train, X_test, y_train, y_test, preprocessor):
    """Tuning qilingan modelni to'liq o'qitish va test to'plamida baholash."""
    
    model_class = ALL_MODELS[model_name]
    
    
    if model_name in ["RandomForestRegressor", "KNeighborsRegressor"] and not GPU_AVAILABLE:
        if 'n_jobs' in best_params: best_params['n_jobs'] = -1
        
    if 'random_state' not in best_params: best_params['random_state'] = 42

    
    model_instance = model_class(**best_params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    return r2, rmse, training_time, pipeline, best_params


def run_tuning_comparison():
    
    
    logger.info("="*50)
    logger.info("TUNING JARAYONI BOSHLANDI (6 TA ALGORITM: Optuna + GPU/CPU).")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(current_dir, ENGINEERED_PATH) 
    
    if not os.path.exists(df_path):
        logger.error(f"XATO: Engineered data fayli '{df_path}' topilmadi. Avval 'run.py' ni ishga tushirib, 2/3-bosqichlarni bajaring.")
        return None

    df_engineered = pd.read_csv(df_path)
    df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info(f"Engineered Dataframes hajmi: {df_engineered.shape}")

    X = df_engineered.drop(columns=[TARGET_COL])
    y = df_engineered[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = create_pipeline(X_train)
    
    results_list = []
    best_model_info = {'R2': -np.inf, 'Pipeline': None, 'Model': None}
    best_params_to_save = {}

    
   
    for model_name, model_class in ALL_MODELS.items():
        start_time_total = time.time()
        
        
        best_params, _ = run_optuna_tuning(model_name, X_train, y_train, preprocessor, n_trials=50) 
        
        
        r2, rmse, t_time, pipeline, final_params = evaluate_final_model(
            model_name, best_params, X_train, X_test, y_train, y_test, preprocessor
        )
        
        total_time = time.time() - start_time_total
        logger.info(f"  > {model_name} (TUNED) | R2: {r2:.4f}, RMSE: {rmse:.0f}, Jami Vaqt: {total_time:.1f}s")
        
        results_list.append({
            'Model': f"{model_name} (Tuned)",
            'Data': 'Engineered',
            'R2': r2,
            'RMSE': rmse,
            'TrainingTime': t_time,
            'TotalTime': total_time 
        })
        
       
        best_params_to_save[model_name] = final_params

        
        if r2 > best_model_info['R2']:
            best_model_info.update({
                'Model': f"{model_name} (Tuned)", 'R2': r2, 'RMSE': rmse, 'Pipeline': pipeline
            })


    results_df = pd.DataFrame(results_list)
    plot_results(results_df)

    
    if best_model_info['Pipeline']:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        params_path = os.path.join(models_dir, 'best_params_tuned.json')
        model_path = os.path.join(models_dir, 'best_tuned_model.joblib')

        os.makedirs(models_dir, exist_ok=True)
        
       
        joblib.dump(best_model_info['Pipeline'], model_path)
        logger.info(f"[SUCCESS] Eng yaxshi TUNED Model Pipeline '{model_path}' ga saqlandi.")
        
        
        def convert_to_serializable(obj):
            
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        serializable_params = {k: {pk: convert_to_serializable(pv) for pk, pv in v.items()} 
                               for k, v in best_params_to_save.items()}
        
        with open(params_path, 'w') as f:
            json.dump(serializable_params, f, indent=4)
        logger.info(f"[SUCCESS] Barcha 6 ta modelning tuning natijalari '{params_path}' ga saqlandi (Stacking uchun).")

        logger.info("-" * 40)
        logger.info("YAKUNIY NATIJA: Giperparametr Tuningi Muvaffaqiyatli.")
        logger.info(f"Eng yaxshi TUNED model: {best_model_info['Model']}")
        logger.info(f"R2 Balli: {best_model_info['R2']:.4f}, RMSE: {best_model_info['RMSE']:.0f}")
        logger.info("-" * 40)
        return True
    
    return False

def plot_results(results_df: pd.DataFrame):
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    results_df_sorted = results_df.sort_values(by='R2', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    
    sns.barplot(x='Model', y='R2', data=results_df_sorted, ax=axes[0], palette='viridis')
    axes[0].set_title("Model Taqqoslash: R2 Balli (Barcha 6 Tuned Modellar)")
    axes[0].tick_params(axis='x', rotation=15)
    
    
    sns.barplot(x='Model', y='TotalTime', data=results_df_sorted, ax=axes[1], palette='plasma')
    axes[1].set_title("Modelni O'qitishga Ketgan Vaqt (Tuning bilan birga, Soniya)")
    axes[1].tick_params(axis='x', rotation=15)

    
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.4f', label_type='edge', fontsize=9)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1fs', label_type='edge', fontsize=9)
    
    plt.suptitle("Giperparametr Tuningidan Keyingi Model Taqqoslash Tahlili (Engineered Data)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(reports_dir, 'model_comparison_tuned.png')
    plt.savefig(plot_path)
    logger.info(f"[SUCCESS] Tuning natijalari grafigi '{plot_path}' ga saqlandi.")
    plt.close()


if __name__ == '__main__':
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    run_tuning_comparison()