

import pandas as pd
import numpy as np
from .utils import setup_logger 

logger = setup_logger('PREPROCESSING_MODULE')

def drop_irrelevant_features(df):
    """Ahamiyatsiz ustunlarni olib tashlaydi."""
    logger.info("3. Ahamiyatsiz ustunlarni olib tashlash boshlandi.")
    cols_to_drop = [
        'dateCrawled', 'dateCreated', 'lastSeen', 'seller', 'offerType', 
        'nrOfPictures', 'monthOfRegistration', 'name', 'index' 
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    logger.info(f"Hozirgi ustunlar soni: {df.shape[1]}")
    return df

def filter_outliers(df):
    """Outlierlarni filtrlash."""
    logger.info("1. Outlierlarni filtrlash boshlandi.")
    initial_rows = df.shape[0]
    
    
    df = df[df['price'].between(100, 100000)]
    df = df[df['powerPS'].between(50, 500)]
    df = df[df['yearOfRegistration'].between(1950, 2016)]

    removed_rows = initial_rows - df.shape[0]
    logger.info(f"Jami olib tashlangan qatorlar (outlierlar): {removed_rows}")
    return df

def feature_engineer_basic(df): 
    """Avtomobil yoshini yaratish va 'yearOfRegistration'ni o'chirish."""
    logger.info("2. Feature Engineering: Avtomobil yoshini (FE_CarAge) yaratish boshlandi.")
    CURRENT_YEAR = 2016 
    df['FE_CarAge'] = CURRENT_YEAR - df['yearOfRegistration'] 
    df = df.drop(columns=['yearOfRegistration'])
    logger.info("   - FE_CarAge ustuni muvaffaqiyatli yaratildi.")
    return df

def handle_missing_values(df):
    """Yetishmayotgan qiymatlarni (NaN) to'ldirish (Imputation)."""
    logger.info("4. Yetishmayotgan qiymatlarni (NaN) to'ldirish boshlandi.")

    df['notRepairedDamage'].fillna('nein', inplace=True)
    cols_to_fill_unbekannt = ['vehicleType', 'fuelType', 'model', 'gearbox'] 
    for col in cols_to_fill_unbekannt:
        df.loc[:, col].fillna('unbekannt', inplace=True)
    
    if 'postalCode' in df.columns:
        df['postalCode'].fillna(df['postalCode'].mode()[0], inplace=True) 
    if 'brand' in df.columns:
        df['brand'].fillna('unbekannt', inplace=True)

    if df.isnull().sum().sum() > 0:
        logger.warning("To'ldirishdan keyin ham NaN qiymatlar qoldi!")
    else:
        logger.info("Barcha NaN qiymatlar muvaffaqiyatli to'ldirildi.") 
    return df

def handle_skewness(df):
    """price, powerPS, FE_CarAge ustunlariga Log Transformatsiyasini qo'llash."""
    logger.info("5. Skewnessni kamaytirish uchun Log Transformatsiyasi boshlandi.")
    skewed_cols = ['price', 'powerPS', 'FE_CarAge']
    for col in skewed_cols:
        df[col] = np.log1p(df[col])
        logger.info(f"   - {col} ustuniga Log(1+x) transformatsiyasi qo'llanildi.")
    return df

def handle_high_cardinality(df, min_count=2000):
    """model ustunini guruhlash (High Cardinality)."""
    logger.info("6. Yuqori kardinalitetni (model) guruhlash boshlandi.")
    value_counts = df['model'].value_counts()
    rare_models = value_counts[value_counts < min_count].index
    df.loc[:, 'model'] = np.where(df['model'].isin(rare_models), 'other', df['model'])
    logger.info(f"   - Yangi model kategoriyalari soni: {df['model'].nunique()}")
    return df


def preprocess_data(df):
    """Asosiy Preprocessing jarayonini boshqarish funksiyasi."""
    logger.info("="*50)
    logger.info("ASOSIY PREPROCESSING JARAYONI BOSHLANDI (Cleaning & Log-Transforms).") 
    
    df = filter_outliers(df)
    df = feature_engineer_basic(df) 
    df = drop_irrelevant_features(df)
    df = handle_missing_values(df)
    df = handle_skewness(df) 
    df = handle_high_cardinality(df)

    logger.info(f"Processed Dataframe hajmi: {df.shape}")
    logger.info("ASOSIY PREPROCESSING MUVAFFAQIYATLI YAKUNLANDI. (Scaling/Encoding keyinroq Pipeline'da bo'ladi).")
    logger.info("="*50)
    
    return df