import re
import logging
import sys

# إعداد التسجيل مع ترميز UTF-8
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# معالجة النصوص باللغة العربية
def preprocess_text(text):
    """تنظيف النص وتحويله إلى صيغة موحدة"""
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()

# استخراج المصطلحات الطبية
def extract_medical_terms(text):
    """استخراج الأعراض والمصطلحات الطبية من النص"""
    symptom_map = {
        'ألم صدر': 'chest_pain', 'وجع صدر': 'chest_pain',
        'صعوبة تنفس': 'breathlessness', 'ضيق تنفس': 'breathlessness',
        'تعب': 'fatigue', 'إرهاق': 'fatigue',
        'صداع': 'headache', 'ألم رأس': 'headache',
        'دوخة': 'dizziness', 'دوار': 'dizziness',
        'صعوبة المشي': 'DifficultyWalking', 'مشكلة المشي': 'DifficultyWalking',
        'صعوبة التركيز': 'DifficultyConcentrating', 'مشكلة التركيز': 'DifficultyConcentrating',
        'فقدان السمع': 'DeafOrHardOfHearing', 'صعوبة السمع': 'DeafOrHardOfHearing',
        'مشاكل الرؤية': 'BlindOrVisionDifficulty', 'صعوبة الرؤية': 'BlindOrVisionDifficulty',
        'التدخين': 'SmokerStatus', 'مدخن': 'SmokerStatus',
        'شرب الكحول': 'AlcoholDrinkers', 'استهلاك الكحول': 'AlcoholDrinkers',
        'كوفيد': 'CovidPos', 'كورونا': 'CovidPos'
    }
    
    terms = {}
    text = preprocess_text(text)
    for ar_term, en_term in symptom_map.items():
        if ar_term in text:
            terms[en_term] = 1
            logging.debug(f"تم اكتشاف العرض: {ar_term} -> {en_term}")
    
    # استخراج عدد أيام الصحة البدنية أو العقلية
    days_match = re.search(r'(\d+)\s*(يوم|أيام)\s*(صحة بدنية|صحة عقلية)', text)
    if days_match:
        days = int(days_match.group(1))
        health_type = days_match.group(3)
        if health_type == 'صحة بدنية':
            terms['PhysicalHealthDays'] = days
        elif health_type == 'صحة عقلية':
            terms['MentalHealthDays'] = days
    
    return terms

# استخراج الديموغرافيا
def extract_demographics(text):
    """استخراج المعلومات الديموغرافية مثل العمر والجنس"""
    demographics = {}
    age_match = re.search(r'(\d+)\s*(سنة|عام|سنوات)', text)
    if age_match:
        demographics['Age'] = int(age_match.group(1))
    if 'ذكر' in text or 'رجل' in text:
        demographics['Sex'] = 'Male'
    elif 'أنثى' in text or 'امرأة' in text or 'بنت' in text:
        demographics['Sex'] = 'Female'
    return demographics

# استخراج الأعراض
def extract_symptoms(text, feature_names, FEATURE_MAPPING):
    """استخراج الأعراض والمعلومات الديموغرافية من النص"""
    try:
        symptoms = {feat: None for feat in feature_names}
        
        # استخراج المصطلحات الطبية
        medical_terms = extract_medical_terms(text)
        for term, value in medical_terms.items():
            if term in feature_names:
                symptoms[term] = value
                logging.info(f"تم استخراج الميزة الطبية: {term} = {value}")
        
        # استخراج المعلومات الديموغرافية
        demographics = extract_demographics(text)
        
        # معالجة العمر
        if 'Age' in demographics:
            age = int(demographics['Age'])
            symptoms['AgeCategory'] = None
            min_diff = float('inf')
            
            for cat, val in FEATURE_MAPPING['AgeCategory'].items():
                current_diff = abs(age - val)
                if current_diff < min_diff:
                    min_diff = current_diff
                    symptoms['AgeCategory'] = val
            
            logging.info(f"تم تعيين فئة العمر {age} → {symptoms['AgeCategory']}")
        
        # معالجة الجنس
        if 'Sex' in demographics:
            sex_input = demographics['Sex'].lower()
            symptoms['Sex'] = next(
                (v for k, v in FEATURE_MAPPING['Sex'].items() if k.lower() in sex_input),
                0
            )
            logging.info(f"تم استخراج الجنس: {demographics['Sex']} → {symptoms['Sex']}")
        
        # استخراج الأرقام
        numeric_features = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 
                          'HeightInMeters', 'WeightInKilograms', 'BMI']
        
        for feat in numeric_features:
            if feat in feature_names:
                num_match = re.search(rf'(\d+\.?\d*)\s*{feat}', text, re.IGNORECASE)
                if num_match:
                    symptoms[feat] = FEATURE_MAPPING[feat](num_match.group(1))
                    logging.info(f"تم استخراج القيمة الرقمية: {feat} = {symptoms[feat]}")
        
        # تعيين قيم افتراضية للميزات المفقودة
        for feat in feature_names:
            if symptoms[feat] is None and feat in ['HeightInMeters', 'WeightInKilograms', 'BMI']:
                symptoms[feat] = FEATURE_MAPPING[feat]("")
        
        logging.debug(f"الأعراض المستخرجة: { {k:v for k,v in symptoms.items() if v is not None} }")
        return symptoms
        
    except Exception as e:
        logging.error(f"خطأ في استخراج الأعراض: {str(e)}", exc_info=True)
        default_values = {feat: 0 for feat in feature_names}
        default_values.update({
            'AgeCategory': 27,
            'Sex': 0,
            'GeneralHealth': 2
        })
        return default_values