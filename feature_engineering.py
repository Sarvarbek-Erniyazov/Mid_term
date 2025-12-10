

import pandas as pd
import numpy as np
import os 
from .utils import setup_logger 

logger = setup_logger('FEATURE_ENGINEERING_MODULE')

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    
    logger.info("="*50)
    logger.info("1. Chuqur Feature Engineering boshlandi (Jami 9 ta feature).")
    
    
    avg_price_brand = df.groupby('brand')['price'].transform('mean')
    df['FE_AvgLogPricePerBrand'] = avg_price_brand
    logger.info("   - FE_AvgLogPricePerBrand ustuni yaratildi.")

    
    avg_power_model = df.groupby('model')['powerPS'].transform('mean')
    df['FE_AvgLogPowerPerModel'] = avg_power_model
    logger.info("   - FE_AvgLogPowerPerModel ustuni yaratildi.")

    
    model_counts = df['model'].value_counts(normalize=True).to_dict()
    df['FE_ModelPopularity'] = df['model'].map(model_counts)
    logger.info("   - FE_ModelPopularity ustuni yaratildi.")
    
    
    df['FE_IsManual'] = np.where(df['gearbox'] == 'manuell', 1, 0)
    logger.info("   - FE_IsManual (Binar) ustuni yaratildi.")

    
    df['FE_HasDamage'] = np.where(df['notRepairedDamage'] == 'ja', 1, 0)
    logger.info("   - FE_HasDamage (Binar) ustuni yaratildi.")
    
    
    df['FE_RegionPrefix'] = df['postalCode'].astype(str).str[:2]
    df.drop(columns=['postalCode'], inplace=True)
    logger.info("   - FE_RegionPrefix (2 xonali pochta kodi) ustuni yaratildi.")

    
    
    df['FE_AgePriceRatio'] = df['FE_CarAge'] / df['price']
    logger.info("   - FE_AgePriceRatio (Yosh/Narx nisbati) ustuni yaratildi.")
    
    
    df['FE_PowerAgeInteraction'] = df['powerPS'] * df['FE_CarAge']
    logger.info("   - FE_PowerAgeInteraction ustuni yaratildi.")
    
    
    mean_power = df['powerPS'].mean()
    mean_price = df['price'].mean()
    df['FE_IsHighEnd'] = np.where(
        (df['powerPS'] > mean_power) & (df['price'] > mean_price), 
        1, 0
    )
    logger.info("   - FE_IsHighEnd (Binar) ustuni yaratildi.")

    
    
    if df.isnull().sum().any() or np.isinf(df.select_dtypes(include=np.number)).sum().sum() > 0:
        logger.warning("FE jarayonida NaN yoki Inf qiymatlar topildi. Tushirilmoqda.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        logger.warning(f"  - Dataframe hajmi: {df.shape}")

    logger.info(f"Yakuniy Engineered Dataframe hajmi: {df.shape}")
    logger.info("CHUQUQ FEATURE ENGINEERING MUVAFFAQIYATLI YAKUNLANDI.")
    logger.info("="*50)
    
    return df