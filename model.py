import os
import pandas as pd
import numpy as np
import joblib
import logging
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import time

# إعداد التسجيل
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# إعداد ملفات البيانات
DATASET_FILE = "D:/chat2/2022/heart_2022_with_nans.csv"

# قاموس أسماء الأمراض/الحالات الصحية
NUMBER_TO_DISEASE = {
    0: 'نوبة قلبية',
    1: 'الذبحة الصدرية',
    2: 'السكتة الدماغية',
    3: 'الربو',
    4: 'سرطان الجلد',
    5: 'مرض الانسداد الرئوي المزمن',
    6: 'الاكتئاب',
    7: 'مرض الكلى',
    8: 'التهاب المفاصل',
    9: 'السكري',
    10: 'كوفيد-19'
}

# تحميل ومعالجة البيانات
def load_and_preprocess_data():
    """تحميل ومعالجة مجموعة البيانات مع التحقق من الأخطاء"""
    try:
        if not os.path.exists(DATASET_FILE):
            raise FileNotFoundError(f"ملف البيانات {DATASET_FILE} غير موجود")
        
        df = pd.read_csv(DATASET_FILE)
        logging.info(f"الأعمدة المتاحة في البيانات: {df.columns.tolist()}")
        
        # تعريف الأعمدة
        base_columns = ['Sex', 'AgeCategory']
        condition_columns = [
            'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
            'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
            'HadDiabetes', 'CovidPos'
        ]
        feature_columns = [
            'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays', 'PhysicalActivities',
            'SleepHours', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
            'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
            'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'AlcoholDrinkers',
            'HeightInMeters', 'WeightInKilograms', 'BMI'
        ]
        
        # التحقق من الأعمدة المفقودة
        missing_cols = [col for col in base_columns + condition_columns + feature_columns if col not in df.columns]
        if missing_cols:
            logging.warning(f"أعمدة مفقودة: {missing_cols}")
        
        # معالجة القيم المفقودة
        df.fillna({
            'Sex': df['Sex'].mode()[0] if 'Sex' in df else 'Female',
            'AgeCategory': df['AgeCategory'].mode()[0] if 'AgeCategory' in df else 'Age 25 to 29',
            'GeneralHealth': df['GeneralHealth'].mode()[0] if 'GeneralHealth' in df else 'Good',
            'PhysicalHealthDays': 0,
            'MentalHealthDays': 0,
            'PhysicalActivities': 'No',
            'SleepHours': df['SleepHours'].median() if 'SleepHours' in df else 7,
            'HeightInMeters': df['HeightInMeters'].median() if 'HeightInMeters' in df else 1.70,
            'WeightInKilograms': df['WeightInKilograms'].median() if 'WeightInKilograms' in df else 70.0,
            'BMI': df['BMI'].median() if 'BMI' in df else 22.0,
            **{col: 'No' for col in condition_columns + ['DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 
                                                         'DifficultyConcentrating', 'DifficultyWalking', 
                                                         'DifficultyDressingBathing', 'DifficultyErrands', 
                                                         'AlcoholDrinkers'] if col in df},
            **{col: df[col].mode()[0] for col in ['SmokerStatus', 'ECigaretteUsage'] if col in df}
        }, inplace=True)
        
        # تنظيف القيم غير الصالحة في الأعمدة الثنائية
        binary_columns = condition_columns + ['PhysicalActivities', 'AlcoholDrinkers', 
                                             'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 
                                             'DifficultyConcentrating', 'DifficultyWalking', 
                                             'DifficultyDressingBathing', 'DifficultyErrands']
        for col in binary_columns:
            if col in df:
                df[col] = df[col].apply(lambda x: x if x in ['Yes', 'No'] else 'No')
                df[col] = df[col].map({'Yes': 1, 'No': 0}).astype(int)
        
        # ترميز القيم
        if 'Sex' in df:
            df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0}).astype(int)
        
        # ترميز GeneralHealth
        general_health_map = {'Excellent': 4, 'Very good': 3, 'Good': 2, 'Fair': 1, 'Poor': 0}
        if 'GeneralHealth' in df:
            df['GeneralHealth'] = df['GeneralHealth'].map(general_health_map).fillna(2).astype(int)
        
        # ترميز AgeCategory
        age_map = {
            'Age 18 to 24': 20, 'Age 25 to 29': 27, 'Age 30 to 34': 32, 'Age 35 to 39': 37,
            'Age 40 to 44': 42, 'Age 45 to 49': 47, 'Age 50 to 54': 52, 'Age 55 to 59': 57,
            'Age 60 to 64': 62, 'Age 65 to 69': 67, 'Age 70 to 74': 72, 'Age 75 to 79': 77,
            'Age 80 or older': 82
        }
        if 'AgeCategory' in df:
            df['AgeCategory'] = df['AgeCategory'].map(age_map).fillna(27).astype(int)
        
        # ترميز SmokerStatus و ECigaretteUsage
        smoker_map = {
            'Never smoked': 0, 'Former smoker': 1, 
            'Current smoker - now smokes every day': 2, 
            'Current smoker - now smokes some days': 2
        }
        ecig_map = {
            'Never used e-cigarettes in my entire life': 0, 'Not at all (right now)': 0, 
            'Use them some days': 1, 'Use them every day': 2
        }
        if 'SmokerStatus' in df:
            df['SmokerStatus'] = df['SmokerStatus'].map(smoker_map).fillna(0).astype(int)
        if 'ECigaretteUsage' in df:
            df['ECigaretteUsage'] = df['ECigaretteUsage'].map(ecig_map).fillna(0).astype(int)
        
        # التحقق من القيم المفقودة مرة أخرى
        if df.isna().any().any():
            logging.warning("تم العثور على قيم مفقودة بعد المعالجة، يتم ملؤها بـ 0")
            df.fillna(0, inplace=True)
        
        logging.info(f"عدد الصفوف: {len(df)}")
        logging.info(f"عدد الميزات: {len(feature_columns)}")
        logging.info(f"توزيع الحالات الصحية:\n{df[condition_columns].sum()}")
        
        return df, feature_columns, condition_columns
    
    except FileNotFoundError as fnf_error:
        logging.error(f"خطأ في العثور على الملف: {fnf_error}")
        return None, None, None
    except pd.errors.EmptyDataError:
        logging.error("ملف البيانات فارغ")
        return None, None, None
    except Exception as e:
        logging.error(f"خطأ غير متوقع في معالجة البيانات: {str(e)}", exc_info=True)
        return None, None, None

# تدريب النموذج وحفظه
def train_and_save_model():
    """تدريب النماذج وحفظها مع التحقق من الأخطاء"""
    data, feature_columns, condition_columns = load_and_preprocess_data()
    if data is None:
        logging.error("فشل تحميل البيانات، لا يمكن متابعة التدريب")
        return False

    try:
        models = {}
        performance_metrics = {}

        for condition in condition_columns:
            logging.info(f"تدريب نموذج لـ {condition}")
            X = data[feature_columns]
            y = data[condition]

            for col in X.columns:
                if X[col].dtype.name == 'object':
                    logging.warning(f"تحويل العمود {col} إلى نوع عددي")
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            if X.isna().any().any():
                logging.warning("تم العثور على قيم مفقودة في البيانات، يتم ملؤها بـ 0")
                X = X.fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            min_class_count = pd.Series(y_train).value_counts().min()
            k_neighbors = min(5, max(1, min_class_count - 1))
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )

            xgb_model = XGBClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                eval_metric='aucpr',
                random_state=42
            )

            voting_model = VotingClassifier(
                estimators=[('rf', rf_model), ('xgb', xgb_model)],
                voting='soft',
                weights=[1.2, 1]
            )

            start_time = time.time()
            voting_model.fit(X_train_res, y_train_res)
            training_time = time.time() - start_time
            logging.info(f"تم تدريب نموذج {condition} في {training_time:.2f} ثانية")

            try:
                rf_model.fit(X_train_res, y_train_res)
            except Exception as rf_error:
                logging.warning(f"⚠️ لم يتمكن من تدريب RandomForest لـ {condition}: {str(rf_error)}")
                rf_model = None

            y_probs = voting_model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            optimal_threshold = thresholds[np.argmax(tpr - fpr)]
            y_pred = (y_probs >= optimal_threshold).astype(int)

            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_probs)
            avg_precision = average_precision_score(y_test, y_probs)

            print(f"\n===== تقييم نموذج {NUMBER_TO_DISEASE[condition_columns.index(condition)]} =====")
            print(f"الدقة: {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Optimal Threshold: {optimal_threshold:.4f}")
            print(pd.DataFrame(report).transpose().to_string())
            print("="*50 + "\n")

            models[condition] = voting_model
            performance_metrics[condition] = {
                'classification_report': report,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'average_precision': avg_precision,
                'optimal_threshold': optimal_threshold,
                'feature_importances': dict(zip(feature_columns, rf_model.feature_importances_)) if rf_model else {}
            }

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=['لا', 'نعم'],
                        yticklabels=['لا', 'نعم'])
            plt.title(f"مصفوفة الارتباك - {NUMBER_TO_DISEASE[condition_columns.index(condition)]}")
            plt.xlabel("التنبؤات")
            plt.ylabel("القيم الحقيقية")
            plt.tight_layout()
            plt.savefig(f"confusion_matrix_{condition}.png", dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {NUMBER_TO_DISEASE[condition_columns.index(condition)]}')
            plt.legend(loc="lower right")
            plt.savefig(f"roc_curve_{condition}.png", dpi=300, bbox_inches='tight')
            plt.close()

        model_components = {
            'models': models,
            'feature_names': feature_columns,
            'condition_columns': condition_columns,
            'performance_metrics': performance_metrics,
            'optimal_thresholds': {cond: metrics['optimal_threshold'] for cond, metrics in performance_metrics.items()}
        }

        joblib.dump(model_components, 'optimized_model_with_features.pkl')
        logging.info("تم حفظ جميع مكونات النموذج المحسن بنجاح في optimized_model_with_features.pkl")

        return True

    except Exception as e:
        logging.error(f"حدث خطأ غير متوقع أثناء التدريب: {str(e)}", exc_info=True)
        return False

# تحميل مكونات النموذج
def load_model_components():
    """تحميل النماذج المحفوظة مع التحقق من الجودة"""
    try:
        try:
            components = joblib.load('optimized_model_with_features.pkl')
            logging.info("✅ تم تحميل النموذج المحسن بنجاح")
        except FileNotFoundError:
            try:
                components = joblib.load('model_with_features.pkl')
                logging.warning("⚠️ تم تحميل النموذج القديم - يوصى بإعادة التدريب")
            except FileNotFoundError:
                logging.warning("📦 لا يوجد نموذج محفوظ، يتم تدريب نموذج جديد...")
                if train_and_save_model():
                    components = joblib.load('optimized_model_with_features.pkl')
                else:
                    raise FileNotFoundError("فشل تدريب النموذج الجديد")
        
        models = components['models']
        feature_names = components['feature_names']
        condition_columns = components['condition_columns']
        
        if not feature_names or len(feature_names) == 0:
            raise ValueError("قائمة أسماء الميزات فارغة")
        
        for condition in condition_columns:
            if condition not in models or not hasattr(models[condition], 'predict'):
                logging.error(f"❌ نموذج {condition} غير صالح")
                raise ValueError(f"نموذج {condition} غير صالح")
        
        if 'optimal_thresholds' in components:
            for condition, model in models.items():
                if condition in components['optimal_thresholds']:
                    model.optimal_threshold = components['optimal_thresholds'][condition]
        
        if 'performance_metrics' in components:
            for condition, metrics in components['performance_metrics'].items():
                disease_name = NUMBER_TO_DISEASE.get(condition_columns.index(condition), condition)
                logging.info(f"\n📊 أداء نموذج {disease_name}:")
                logging.info(f"الدقة: {metrics.get('accuracy', 0):.2f}")
                logging.info(f"التذكر (Recall): {metrics.get('recall', 0):.2f}")
                logging.info(f"الدقة (Precision): {metrics.get('precision', 0):.2f}")
                if 'optimal_threshold' in metrics:
                    logging.info(f"العتبة المثلى: {metrics['optimal_threshold']:.4f}")
        
        logging.info(f"✅ تم تحميل النماذج بنجاح مع {len(feature_names)} ميزة و {len(condition_columns)} حالة صحية")
        return models, feature_names, condition_columns
    
    except Exception as e:
        logging.error(f"❌ خطأ أثناء تحميل النموذج: {e}", exc_info=True)
        return None, [], []