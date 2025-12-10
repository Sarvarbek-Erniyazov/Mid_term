# 🚗 Ishlatilgan Avtomobillar Yurgan Masofasini Bashoratlash (Used Car Kilometer Prediction)

**Loyiha Maqsadi:** Data Science va Machine Learning usullari yordamida **"kilometer"** (bosib o'tgan masofa) ustunini bashorat qilish orqali, bu ustunning avtomobil xususiyatlari bilan qanchalik bog'liqligini aniqlash.

## 📌 Hozirgi Holat (Tezkor Hisobot)

Loyihaning barcha asosiy bosqichlari (Ma'lumotlarni tozalash, Xususiyatlarni yaratish va Modellarni Tuning qilish) yakunlanish arafasida.

| Bosqich | Holat | Izoh |
| :--- | :--- | :--- |
| **1. Ma'lumotlarni tozalash (Preprocessing)** | ✅ Yakunlandi | Outlierlar filtrlangan, `Log` transformatsiyasi qo'llanilgan, `autos_processed.csv` tayyor. |
| **2. Xususiyatlarni yaratish (Feature Engineering)** | ✅ Yakunlandi | 9 ta yangi xususiyat yaratildi (Masalan: `FE_CarAge`, `FE_AvgLogPricePerBrand`, `FE_PowerAgeInteraction`). `autos_engineered.csv` tayyor. |
| **3. Modellarni o'qitish (Base Models)** | ⚠️ Davom etmoqda | **6 ta model** (RF, GBT, Ridge, Lasso, KNN, SVR) bazaviy parametrlar bilan Engineered Data ustida sinovdan o'tkazildi. |
| **4. Giperparametr Tuningi (Optuna)** | ⏳ Hozirgi vazifa | **Barcha 6 ta model** Optuna yordamida tuning qilinmoqda (GPU rejimida ishlash imkoniyati bilan). |
| **5. Yakuniy Model (Stacking)** | 🗓️ Keyingi bosqich | Tuning natijalari asosida **Stacking Regressor** yaratiladi. |

---

## I. 💾 Ma'lumotlarga Ishlov Berish va Injiniring

Ma'lumotlarning sifatini oshirish va modelga kiritiladigan signal kuchini kuchaytirish uchun murakkab bosqichlar bajarildi.

### 1. Ma'lumotlarni Tozalash (Preprocessing)

* **Outlierlarni filtrlash:** `price`, `powerPS` va `yearOfRegistration` ustunlaridagi ehtimoliy g'ayritabiiy qiymatlar olib tashlandi.
* **Log Transformatsiyasi:** `price`, `powerPS` va `FE_CarAge` (Avtomobil yoshi) ustunlarida Skewness (qiyalik)ni kamaytirish uchun $Log(1+x)$ transformatsiyasi qo'llanildi.
* **Yuqori Kardinalitet (Model):** 2000 martadan kam uchraydigan barcha avtomobil modellarini **'other'** deb guruhlash orqali kategoriyalar soni optimallashtirildi.

### 2. Xususiyatlarni Yaratish (Feature Engineering)

Jami **9 ta kuchli xususiyat** qo'shildi, ular `kilometer` (target) ga ta'sir qiluvchi murakkab aloqalarni aks ettiradi:

* **Brend/Model Asosidagi Statistika:**
    * `FE_AvgLogPricePerBrand` (Brend bo'yicha o'rtacha log-narx)
    * `FE_AvgLogPowerPerModel` (Model bo'yicha o'rtacha log-quvvat)
    * `FE_ModelPopularity` (Modelning chastota kodlashi)
* **Vaqtga Bog'liq O'zaro Ta'sirlar:**
    * `FE_CarAge` (Avtomobil yoshi)
    * `FE_AgePriceRatio` (Yosh va Narxning nisbati)
    * `FE_PowerAgeInteraction` (Quvvat * Avtomobil yoshi)
* **Binar va Kategorik Xususiyatlar:**
    * `FE_IsManual`, `FE_HasDamage`, `FE_IsHighEnd` (Yuqori quvvatli/qimmat)
    * `FE_RegionPrefix` (Pochta kodining dastlabki 2 xonasi)

---

## II. 📈 Modellash va Tuning (Keyingi Bosqichlar)

Loyihaning eng muhim qismi — optimal bashorat qilish modelini topish.

### 1. Tuning Qilinayotgan Modellar

Barcha 6 ta tanlangan Regressiya modeli Optuna yordamida giperparametr tuningidan o'tkaziladi.

* `RandomForestRegressor` (GPU bilan tezlatish imkoniyati bilan)
* `GradientBoostingRegressor` (GPU bilan tezlatish imkoniyati bilan)
* `Ridge`
* `Lasso`
* `KNeighborsRegressor`
* `LinearSVR`

> **Maqsad:** Har bir model uchun **eng yuqori R2 ballini** beradigan parametrlarni topish.

### 2. Tuning Natijalarini Saqlash

Tuningdan so'ng, natijalar `models/` katalogiga quyidagi tartibda saqlanadi:

| Fayl | Maqsad |
| :--- | :--- |
| **`models/best_params_tuned.json`** | **Barcha 6 ta modelning** optimal parametrlari. **(Stacking bosqichida Metamodellar uchun kirish vazifasini bajaradi)** |
| **`models/best_tuned_model.joblib`** | 6 ta model ichida test to'plamida **eng yaxshi R2 ballini** ko'rsatgan model (Pipeline bilan birga). |
| **`reports/model_comparison_tuned.png`** | Tuning natijalarining vizual hisoboti (R2 va Vaqt bo'yicha). |

### 3. Keyingi Bosqich: Stacking

Tuningdan olingan eng yaxshi 6 ta model (ularning optimal parametrlari bilan) **Stacking Regressor** yaratish uchun asosiy modellar sifatida ishlatiladi. Stacking odatda har bir individual modeldan yuqoriroq ball beradi.

---

## III. 💡 Loyihani Ishga Tushirish

Loyihani to'liq ishga tushirish uchun quyidagi buyruqlar ketma-ket bajarilishi kerak:

```bash
# Loyihaning asosiy katalogiga o'ting
cd used_car_km_prediction

# 1. Ma'lumotni tozalash va FE (run.py)
python run.py 

# 2. Modellarni Tuning qilish (alohida tuning.py faylida)
cd src
python tuning.py
