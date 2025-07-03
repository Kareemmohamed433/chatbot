import re
import logging
import sys
import pandas as pd
import requests
import uuid
from model import load_model_components, NUMBER_TO_DISEASE
from rl_agent import RLAgent
from datetime import datetime

# إعداد التسجيل مع ترميز UTF-8
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# تحميل النماذج
try:
    models, feature_names, condition_columns = load_model_components()
    if not feature_names:
        logging.error("❌ فشل في الحصول على قائمة الميزات، سيتم إنهاء التطبيق")
        sys.exit(1)
    logging.info("✔ تم تهيئة جميع النماذج بنجاح وجاهزة للاستخدام")
except Exception as e:
    logging.error(f"❌ خطأ فادح في تحميل النماذج: {str(e)}", exc_info=True)
    sys.exit(1)

# تهيئة OpenRouter
OPENROUTER_API_KEY = "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"

# ==============================================
# RESPONSE TEMPLATES
# ==============================================

RESPONSE_TEMPLATES = {
    "greetings": [
        ("مرحبًا", "مرحبًا بك! 😊 كيف يمكنني مساعدتك اليوم؟"),
        ("السلام عليكم", "وعليكم السلام ورحمة الله 🌹 كيف تصف صحتك العامة اليوم؟"),
        ("أهلًا", "أهلًا وسهلًا! 💙 هل لديك أي استفسار طبي؟"),
        ("مساء الخير", "مساء الخير والنور! 🌙 كيف تشعر اليوم؟"),
        ("صباح الخير", "صباح النور! ☀️ كيف هو شعورك هذا الصباح؟"),

        ("كيف حالك؟", "أنا بخير، شكرًا لسؤالك! 🤖 أنا هنا لمساعدتك في أي استفسار طبي."),
        ("هل يمكنك مساعدتي؟", "بالطبع! 💪 هل يمكنك وصف الأعراض التي تشعر بها؟"),
        ("أحتاج مساعدة طبية", "أنا هنا لمساعدتك. 🏥 من فضلك صف حالتك أو اذكر الأعراض.")
    ],
    
    "presence_confirmation": [
        ("انت هنا لي", "نعم، أنا هنا لأجلك دائمًا. 💙 هل لديك أي استفسار؟"),
        ("انت هنا لي؟", "بالطبع! أنا هنا لمساعدتك في أي وقت. 😊"),
        ("هل ما زلت هنا؟", "نعم، أنا هنا وأستمع إليك. 💙 هل تريد إضافة أي أعراض؟"),
        ("هل يمكنك مساعدتي الآن؟", "نعم، مست castles لمساعدتك الآن. 💪 من فضلك صف مشكلتك.")
    ],
    
    "general_health": [
        ("كيف أعرف إذا كنت مريضًا؟", 
         "علامات المرض تشمل:\n"
         "- حرارة مرتفعة (أكثر من 38°C)\n"
         "- ألم مستمر\n"
         "- تعب غير عادي\n"
         "- تغير في الشهية أو الوزن\n"
         "إذا استمرت الأعراض أكثر من 3 أيام، استشر طبيبًا."),
         
        ("ما هي أعراض البرد؟",
         "أعراض البرد الشائعة:\n"
         "- سيلان الأنف\n"
         "- عطس\n"
         "- احتقان الحلق\n"
         "- سعال خفيف\n"
         "- صداع خفيف\n"
         "عادة ما تتحسن خلال 7-10 أيام."),
         
        ("كيف أتحسن من الإنفلونزا؟",
         "للتخفيف من الإنفلونزا:\n"
         "1. الراحة الكافية\n"
         "2. شرب السوائل الدافئة\n"
         "3. تناول مسكنات الألم (مثل باراسيتامول)\n"
         "4. استخدام كمادات دافئة للصداع\n"
         "5. تناول فيتامين C\n"
         "إذا تفاقمت الأعراض، راجع الطبيب."),
         
        ("متى يجب أن أذهب إلى الطبيب؟",
         "اذهب للطبيب فورًا إذا واجهت:\n"
         "- صعوبة في التنفس\n"
         "- ألم صدر شديد\n"
         "- ارتفاع حرارة فوق 39°C\n"
         "- تشوش ذهني\n"
         "- قيء أو إسهال مستمر\n"
         "- طفح جلدي مع حرارة")
    ],
    
    "symptoms": {
        "أشعر بألم في صدري": "ألم الصدر قد يكون بسبب:\n"
                            "- مشاكل قلبية (إذا كان الألم مفاجئًا وشديدًا)\n"
                            "- حرقة المعدة\n"
                            "- شد عضلي\n"
                            "- قلق\n"
                            "إذا استمر الألم أكثر من 15 دقيقة مع تعرق أو غثيان، اتصل بالإسعاف.",
                            
        "لدي حرارة عالية": "لخفض الحرارة:\n"
                          "1. استخدم خافض حرارة (باراسيتامول)\n"
                          "2. اشرب الكثير من السوائل\n"
                          "3. استخدم كمادات ماء فاتر\n"
                          "4. ارتد ملابس خفيفة\n"
                          "إذا تجاوزت 39°C أو استمرت أكثر من 3 أيام، راجع الطبيب.",
                          
        "أعاني من صداع شديد": "للتخفيف من الصداع:\n"
                             "- استرح في غرفة مظلمة\n"
                             "- ضع كمادات باردة على الجبين\n"
                             "- تجنب الضوضاء والأضواء الساطعة\n"
                             "- اشرب الماء بكميات كافية\n"
                             "إذا صاحب الصداع تقيؤ أو تشوش الرؤية، استشر طبيبًا.",
                             
        "أشعر بالتعب دائمًا": "أسباب التعب المستمر:\n"
                             "- نقص النوم\n"
                             "- فقر الدم\n"
                             "- التوتر والقلق\n"
                             "- سوء التغذية\n"
                             "- بعض الأمراض المزمنة\n"
                             "جرب تنظيم مواعيد النوم وتحسين النظام الغذائي."
    },
    
    "chronic_conditions": {
        "ما هي أعراض السكري؟": "أعراض السكري:\n"
                              "- العطش الشديد\n"
                              "- التبول المتكرر\n"
                              "- الجوع المستمر\n"
                              "- فقدان الوزن غير المبرر\n"
                              "- التعب الشديد\n"
                              "- تشوش الرؤية\n"
                              "إذا لاحظت هذه الأعراض، قم بفحص السكر.",
                              
        "كيف أتعامل مع ضغط الدم المرتفع؟": "لإدارة ضغط الدم المرتفع:\n"
                                        "- قلل الملح في الطعام\n"
                                        "- مارس الرياضة بانتظام\n"
                                        "- تجنب التوتر\n"
                                        "- أقلع عن التدخين\n"
                                        "- تناول الأدوية بانتظام إذا وصفها الطبيب\n"
                                        "- راقب ضغطك دوريًا",
                                        
        "هل الربو خطير؟": "الربو مرض مزمن لكن يمكن التحكم فيه:\n"
                         "- استخدام البخاخات الوقائية\n"
                         "- تجنب المحفزات (الغبار، الدخان)\n"
                         "- حمل البخاخ الإسعافي دائمًا\n"
                         "- المتابعة مع طبيب الصدر\n"
                         "في حالة النوبة الشديدة (صعوبة كلام، زرقة)، اذهب للمستشفى فورًا."
    },
    
    "mental_health": {
        "أشعر بالقلق دائمًا": "للتغلب على القلق:\n"
                             "- تنفس بعمق وببطء\n"
                             "- مارس تمارين الاسترخاء\n"
                             "- نظم وقتك وقلل الضغوط\n"
                             "- تجنب الكافيين الزائد\n"
                             "- تحدث مع مختص إذا استمر القلق",
                             
        "ما هي أعراض الاكتئاب؟": "أعراض الاكتئاب:\n"
                                "- حزن مستمر\n"
                                "- فقدان الاهتمام بالأنشطة\n"
                                "- تغيرات في النوم أو الشهية\n"
                                "- تعب دائم\n"
                                "- صعوبة في التركيز\n"
                                "- أفكار سلبية متكررة\n"
                                "إذا استمرت الأعراض أكثر من أسبوعين، استشر مختصًا.",
                                
        "كيف أتخلص من التوتر؟": "لإدارة التوتر:\n"
                               "- خذ فترات راحة قصيرة\n"
                               "- مارس الرياضة بانتظام\n"
                               "- استخدم تقنيات التنفس\n"
                               "- نظم أولوياتك\n"
                               "- تواصل مع الأصدقاء\n"
                               "- احصل على قسط كاف من النوم"
    },
    
    "medications": {
        "ما هو دواء باراسيتامول؟": "الباراسيتامول:\n"
                                  "- مسكن للألم وخافض للحرارة\n"
                                  "- الجرعة المعتادة: 500-1000 مجم كل 6 ساعات\n"
                                  "- الحد الأقصى: 4000 مجم يوميًا\n"
                                  "- تجنب تناوله مع الكحول\n"
                                  "- مناسب لمعظم الناس لكن استشر الطبيب إذا كنت تعاني من مشاكل في الكبد",
                                  
        "هل المضادات الحيوية آمنة؟": "المضادات الحيوية:\n"
                                    "- تستخدم فقط للعدوى البكتيرية\n"
                                    "- لا تفيد في نزلات البرد أو الإنفلونزا (فيروسية)\n"
                                    "- يجب إكمال الجرعة كاملة حتى مع تحسن الأعراض\n"
                                    "- قد تسبب آثارًا جانبية مثل الإسهال\n"
                                    "- لا تستخدمها دون وصفة طبية",
                                    
        "كيف أستخدم هذا الدواء؟": "للاستخدام الآمن للأدوية:\n"
                                 "- اقرأ النشرة الدوائية\n"
                                 "- التزم بالجرعة المحددة\n"
                                 "- خذ الدواء مع/بدون طعام حسب التعليمات\n"
                                 "- لا تكسر الحبوب إلا إذا ذكر الصيدلي ذلك\n"
                                 "- احفظ الأدوية في مكان مناسب (جاف، بعيد عن الحرارة)"
    },
    
    "emergency": {
        "ماذا أفعل إذا تعرضت لنوبة قلبية؟": "في حالة النوبة القلبية:\n"
                                          "1. اتصل بالإسعاف فورًا\n"
                                          "2. اجلس أو استلق في وضع مريح\n"
                                          "3. امضغ قرص أسبرين (إذا أوصى به الطبيب سابقًا)\n"
                                          "4. لا تقود بنفسك للمستشفى\n"
                                          "5. حاول البقاء هادئًا حتى وصول المساعدة",
                                          
        "كيف أتعامل مع إصابة خطيرة؟": "للإسعافات الأولية للإصابات:\n"
                                     "- توقف عن النزيف بالضغط المباشر\n"
                                     "- لا تحرك الأطراف المكسورة\n"
                                     "- غطي الجروح بضمادة نظيفة\n"
                                     "- في حالة الحروق: استخدم ماء بارد (ليس ثلج)\n"
                                     "- اطلب المساعدة الطبية فورًا للإصابات الخطيرة"
    },
    
    "prevention": {
        "كيف أحمي نفسي من الأمراض؟": "للوقاية من الأمراض:\n"
                                    "- اغسل يديك بانتظام\n"
                                    "- تناول طعامًا صحيًا\n"
                                    "- مارس الرياضة\n"
                                    "- احصل على التطعيمات\n"
                                    "- نم جيدًا\n"
                                    "- تجنب التدخين والكحول\n"
                                    "- تحكم في التوتر",
                                    
        "ما هي الأطعمة الصحية للقلب؟": "أطعمة لصحة القلب:\n"
                                      "- الخضروات الورقية\n"
                                      "- الأسماك الدهنية (السلمون)\n"
                                      "- المكسرات النيئة\n"
                                      "- زيت الزيتون\n"
                                      "- الفواكه الطازجة\n"
                                      "- الحبوب الكاملة\n"
                                      "- قلل من الملح والدهون المشبعة"
    },
    
    "bot_commands": {
        "أعد تشخيصي": "سأعيد تقييم حالتك. هل لديك أعراض جديدة أو تغييرات؟",
        "ما هي آخر النصائح الطبية؟": "أحدث التوصيات الصحية:\n"
                                    "- المشي 30 دقيقة يوميًا\n"
                                    "- شرب 8 أكواب ماء\n"
                                    "- تقليل السكر والملح\n"
                                    "- فحص ضغط الدم والسكر بانتظام\n"
                                    "- إجراء الفحوصات الدورية بعد سن الأربعين",
        "غير لغة المحادثة": "حاليًا أتحدث العربية فقط. هل تريد الاستمرار بالعربية؟"
    }
}

# معالجة النصوص باللغة العربية
def preprocess_text(text):
    """تنظيف النص وتحويله إلى صيغة موحدة"""
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()

# حساب مؤشر كتلة الجسم (BMI)
def calculate_bmi(weight, height):
    """حساب BMI بناءً على الوزن (كجم) والطول (متر)"""
    try:
        if not isinstance(weight, (int, float)) or not isinstance(height, (int, float)):
            raise ValueError("الوزن والطول يجب أن يكونا أرقامًا")
        if weight <= 0 or height <= 0:
            raise ValueError("الوزن والطول يجب أن يكونا أكبر من صفر")
        bmi = weight / (height * height)
        # تقييد BMI ضمن النطاق المعقول (10 إلى 50)
        bmi = max(10, min(50, bmi))
        logging.debug(f"تم حساب BMI: {bmi:.1f} من الوزن {weight} كجم والطول {height} متر")
        return round(bmi, 1)
    except Exception as e:
        logging.warning(f"خطأ في حساب BMI: {str(e)}")
        return 0

# استخراج اسم المرض
def extract_disease_name(text):
    """استخراج اسم المرض من النص باستخدام تعبيرات منتظمة"""
    text = preprocess_text(text)
    diseases = [
        'السكري', 'ارتفاع ضغط الدم', 'القلب', 'الربو', 'التهاب المفاصل',
        'الاكتئاب', 'القلق', 'السرطان', 'الكوليسترول', 'السمنة', 'مرض الانسداد الرئوي المزمن'
    ]
    for disease in diseases:
        if disease in text:
            return disease
    return None

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
        
        medical_terms = extract_medical_terms(text)
        for term, value in medical_terms.items():
            if term in feature_names:
                symptoms[term] = value
                logging.info(f"تم استخراج الميزة الطبية: {term} = {value}.")
        
        demographics = extract_demographics(text)
        
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
        
        if 'Sex' in demographics:
            sex_input = demographics['Sex'].lower()
            symptoms['Sex'] = next(
                (v for k, v in FEATURE_MAPPING['Sex'].items() if k.lower() in sex_input),
                0
            )
            logging.info(f"تم استخراج الجنس: {demographics['Sex']} → {symptoms['Sex']}")
        
        numeric_features = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 
                           'HeightInMeters', 'WeightInKilograms']
        
        for feat in numeric_features:
            if feat in feature_names:
                num_match = re.search(rf'(\d+\.?\d*)\s*{feat}', text, re.IGNORECASE)
                if num_match:
                    symptoms[feat] = FEATURE_MAPPING[feat](num_match.group(1))
                    logging.info(f"تم استخراج القيمة الرقمية: {feat} = {symptoms[feat]}")
        
        # حساب BMI إذا كان كل من الطول والوزن متاحين
        if 'HeightInMeters' in symptoms and 'WeightInKilograms' in symptoms:
            height = symptoms['HeightInMeters']
            weight = symptoms['WeightInKilograms']
            if height is not None and weight is not None and height > 0 and weight > 0:
                symptoms['BMI'] = calculate_bmi(weight, height)
                logging.info(f"تم حساب BMI ديناميكيًا: {symptoms['BMI']}")
        
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
            'GeneralHealth': 2,
            'BMI': 0
        })
        return default_values

# إنشاء تعيين الميزات
def generate_feature_mapping(features):
    mapping = {}
    for feature in features:
        try:
            if feature == 'Sex':
                mapping[feature] = {
                    'ذكر': 1, 'رجل': 1, 'male': 1,
                    'أنثى': 0, 'امرأة': 0, 'بنت': 0, 'female': 0
                }
            elif feature == 'GeneralHealth':
                mapping[feature] = {
                    'ممتازة': 4, 'excellent': 4,
                    'جيدة جدًا': 3, 'very good': 3,
                    'جيدة': 2, 'good': 2,
                    'متوسطة': 1, 'fair': 1,
                    'ضعيفة': 0, 'poor': 0
                }
            elif feature == 'AgeCategory':
                mapping[feature] = {
                    '18-24': 20, '18 إلى 24': 20,
                    '25-29': 27, '25 إلى 29': 27,
                    '30-34': 32, '30 إلى 34': 32,
                    '35-39': 37, '35 إلى 39': 37,
                    '40-44': 42, '40 إلى 44': 42,
                    '45-49': 47, '45 إلى 49': 47,
                    '50-54': 52, '50 إلى 54': 52,
                    '55-59': 57, '55 إلى 59': 57,
                    '60-64': 62, '60 إلى 64': 62,
                    '65-69': 67, '65 إلى 69': 67,
                    '70-74': 72, '70 إلى 74': 72,
                    '75-79': 77, '75 إلى 79': 77,
                    '80 أو أكثر': 82, '80+': 82
                }
            elif feature == 'SmokerStatus':
                mapping[feature] = {
                    'لم أدخن أبدًا': 0, 'never smoked': 0,
                    'مدخن سابق': 1, 'former smoker': 1,
                    'مدخن يوميًا': 2, 'current smoker - every day': 2,
                    'مدخن أحيانًا': 2, 'current smoker - some days': 2
                }
            elif feature == 'ECigaretteUsage':
                mapping[feature] = {
                    'لم أستخدم أبدًا': 0, 'never used': 0,
                    'لا أستخدم حاليًا': 0, 'not at all': 0,
                    'أستخدم أحيانًا': 1, 'some days': 1,
                    'أستخدم يوميًا': 2, 'every day': 2
                }
            elif feature in ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours']:
                mapping[feature] = lambda x: max(0, min(30, float(x))) if x.replace('.', '', 1).isdigit() else 0
            elif feature in ['HeightInMeters']:
                mapping[feature] = lambda x: float(x) if x.replace('.', '', 1).isdigit() else 1.70
            elif feature in ['WeightInKilograms']:
                mapping[feature] = lambda x: float(x) if x.replace('.', '', 1).isdigit() else 1.0
            elif feature == 'BMI':
                mapping[feature] = lambda x: float(x) if x.replace('.', '', 1).isdigit() else 0
            else:
                mapping[feature] = {
                    'نعم': 1, 'yes': 1, 'يعاني': 1, 'يوجد': 1,
                    'لا': 0, 'no': 0, 'لا يعاني': 0, 'لا يوجد': 0
                }
        except Exception as e:
            logging.error(f"خطأ في إنشاء تعيين للميزة {feature}: {str(e)}")
            mapping[feature] = {'نعم': 1, 'لا': 0}
    return mapping

FEATURE_MAPPING = generate_feature_mapping(feature_names)

# إنشاء الأسئلة المتابعة
def generate_follow_up_questions(features):
    arabic_feature_names = {
        'GeneralHealth': 'الصحة العامة', 'PhysicalHealthDays': 'أيام الصحة البدنية',
        'MentalHealthDays': 'أيام الصحة العقلية', 'PhysicalActivities': 'الأنشطة البدنية',
        'SleepHours': 'ساعات النوم', 'DeafOrHardOfHearing': 'صعوبة السمع',
        'BlindOrVisionDifficulty': 'صعوبة الرؤية', 'DifficultyConcentrating': 'صعوبة التركيز',
        'DifficultyWalking': 'صعوبة المشي', 'DifficultyDressingBathing': 'صعوبة اللباس/الاستحمام',
        'DifficultyErrands': 'صعوبة القيام بالمهام', 'SmokerStatus': 'حالة التدخين',
        'ECigaretteUsage': 'استخدام السجائر الإلكترونية', 'AlcoholDrinkers': 'شرب الكحول',
        'HeightInMeters': 'الطول (متر)', 'WeightInKilograms': 'الوزن (كجم)', 'BMI': 'مؤشر كتلة الجسم',
        'Sex': 'الجنس', 'AgeCategory': 'فئة العمر'
    }
    
    follow_up = {}
    for feature in features:
        if feature in ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms']:
            follow_up[feature] = {
                'text': f"ما هي {arabic_feature_names.get(feature, feature)}؟",
                'options': None,
                'type': 'number'
            }
        elif feature == 'Sex':
            follow_up[feature] = {
                'text': f"ما هو {arabic_feature_names.get(feature, feature)}؟",
                'options': ['ذكر', 'أنثى'],
                'type': 'gender'
            }
        elif feature == 'GeneralHealth':
            follow_up[feature] = {
                'text': f"كيف تصف {arabic_feature_names.get(feature, feature)}؟",
                'options': ['ممتازة', 'جيدة جدًا', 'جيدة', 'متوسطة', 'ضعيفة'],
                'type': 'category'
            }
        elif feature == 'AgeCategory':
            follow_up[feature] = {
                'text': f"ما هي {arabic_feature_names.get(feature, feature)}؟",
                'options': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 أو أكثر'],
                'type': 'category'
            }
        elif feature in ['SmokerStatus', 'ECigaretteUsage']:
            options = (
                ['لم أدخن أبدًا', 'مدخن سابق', 'مدخن يوميًا', 'مدخن أحيانًا'] if feature == 'SmokerStatus'
                else ['لم أستخدم أبدًا', 'لا أستخدم حاليًا', 'أستخدم أحيانًا', 'أستخدم يوميًا']
            )
            follow_up[feature] = {
                'text': f"ما هي {arabic_feature_names.get(feature, feature)}؟",
                'options': options,
                'type': 'category'
            }
        else:
            follow_up[feature] = {
                'text': f"هل تعاني من {arabic_feature_names.get(feature, feature)}؟",
                'options': ['نعم', 'لا'],
                'type': 'binary'
            }
    return follow_up

FOLLOW_UP_QUESTIONS = generate_follow_up_questions(feature_names)
rl_agent = RLAgent(questions=feature_names, diseases=list(NUMBER_TO_DISEASE.values()))

# دالة لمعالجة الاستفسارات العامة
def handle_general_query(query, language='ar'):
    """معالجة الاستفسارات العامة باستخدام القوالب المحددة"""
    query_cleaned = preprocess_text(query)
    logging.debug(f"معالجة الاستفسار العام: {query_cleaned}")
    
    # Check all response templates
    for category, responses in RESPONSE_TEMPLATES.items():
        if isinstance(responses, dict):
            for pattern, response in responses.items():
                if pattern in query_cleaned:
                    return response
        else:
            for pattern, response in responses:
                if pattern in query_cleaned:
                    return response
    
    # If no template matches, use OpenRouter
    prompt = f"""
    المستخدم يسأل: {query}
    أجب باللغة العربية بشكل واضح ومفصل، مع التركيز على تقديم معلومات طبية دقيقة وسهلة الفهم.
    """
    return query_openrouter(prompt)

def query_openrouter(prompt, model="mistralai/mistral-small-3.2-24b-instruct:free", temperature=0.5, max_tokens=500):
    """
    استعلام OpenRouter API للحصول على ردود عامة أو شرح طبي مفصل
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://medical-assistant.com",
        "X-Title": "Medical Assistant"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "أنت مساعد طبي ذكي تقدم معلومات دقيقة وواضحة عن الأمراض باللغة العربية. قدم إجابات مفصلة وسهلة الفهم مع نصائح عملية."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "❌ لم يتم الحصول على رد صحيح من النموذج."
            
    except requests.exceptions.RequestException as e:
        logging.error(f"خطأ في الاتصال بـ OpenRouter: {str(e)}")
        return f"❌ خطأ في الاتصال: {str(e)}"
    except Exception as e:
        logging.error(f"خطأ غير متوقع في OpenRouter: {str(e)}")
        return f"❌ خطأ غير متوقع: {str(e)}"

# معالجة إجابة المستخدم
def process_answer(question, answer, answer_type):
    logging.debug(f"معالجة الإجابة لـ {question}: {answer} (النوع: {answer_type})")
    if answer_type == 'number':
        try:
            num = float(answer)
            if question == 'HeightInMeters' and not (0.5 <= num <= 2.5):
                logging.warning(f"قيمة الطول غير منطقية: {num} متر")
                return None, "يرجى إدخال طول بين 0.5 و2.5 متر."
            if question == 'WeightInKilograms' and not (20 <= num <= 200):
                logging.warning(f"قيمة الوزن غير منطقية: {num} كجم")
                return None, "يرجى إدخال وزن بين 20 و200 كجم."
            if num < 0:
                logging.warning(f"القيمة السلبية غير صالحة لـ {question}: {num}")
                return None, "يرجى إدخال قيمة غير سلبية."
            return num, None
        except ValueError:
            logging.warning(f"إدخال غير صالح لـ {question}: {answer}")
            return None, f"يرجى إدخال قيمة عددية صالحة لـ {question}."
    elif answer_type in ['gender', 'binary', 'category']:
        value = FEATURE_MAPPING.get(question, {}).get(answer, None)
        if value is None:
            logging.warning(f"إجابة غير صالحة لـ {question}: {answer}")
            return None, f"يرجى اختيار إجابة من الخيارات المتاحة: {FOLLOW_UP_QUESTIONS[question]['options']}"
        return value, None
    return 0, None

# تحضير الميزات
def prepare_features(symptoms):
    try:
        features_prepared = {}
        missing_features = []
        
        for key, value in symptoms.items():
            if key in FEATURE_MAPPING:
                mapping = FEATURE_MAPPING[key]
                if callable(mapping):
                    str_value = str(value) if value is not None else '0'
                    features_prepared[key] = mapping(str_value)
                else:
                    features_prepared[key] = mapping.get(value, 0) if value is not None else 0
        
        for feat in feature_names:
            if feat not in features_prepared:
                missing_features.append(feat)
                features_prepared[feat] = (
                    FEATURE_MAPPING[feat]("") if feat in ['HeightInMeters', 'WeightInKilograms', 'BMI']
                    else 0
                )
        
        if missing_features:
            logging.warning(f"الميزات المفقودة: {missing_features}")
        
        ranges = {
            'AgeCategory': (18, 82),
            'PhysicalHealthDays': (0, 30),
            'MentalHealthDays': (0, 30),
            'SleepHours': (0, 24),
            'HeightInMeters': (0.5, 2.5),
            'WeightInKilograms': (20, 200),
            'BMI': (10, 50)
        }
        
        for feature, (min_val, max_val) in ranges.items():
            if feature in features_prepared:
                value = features_prepared.get(feature)
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    features_prepared[feature] = min(max_val, max(min_val, value))
                    logging.warning(f"قيمة غير منطقية لـ {feature}: {value}, تم تعديلها إلى {features_prepared[feature]}")
        
        logging.info(f"تم تحضير {len(features_prepared)} ميزة")
        return features_prepared
        
    except Exception as e:
        logging.error(f"خطأ في تحضير الميزات: {str(e)}", exc_info=True)
        return {feat: 0 for feat in feature_names}

# تحديد السؤال التالي
def determine_next_question(state):
    current_state = rl_agent.get_state(state['symptoms'])
    available_questions = [q for q in feature_names if state['symptoms'].get(q) is None and q != 'BMI']
    if not available_questions:
        logging.debug("لا توجد أسئلة متاحة متبقية")
        return None
    next_question = rl_agent.choose_action(current_state, available_questions)
    logging.debug(f"تم اختيار السؤال التالي: {next_question}")
    return next_question

# إجراء التنبؤ
def make_prediction(features):
    try:
        if not features or not isinstance(features, dict):
            raise ValueError("الميزات غير صالحة")
        
        if not models or not feature_names or not condition_columns:
            raise ValueError("النماذج غير مهيأة")
        
        features_df = pd.DataFrame([features])[feature_names]
        if features_df.isnull().any().any():
            logging.warning("قيم مفقودة في الإدخال، يتم تعبيئتها بصفر")
            features_df = features_df.fillna(0)
        
        predictions = {}
        confidences = {}
        all_probabilities = {}
        detailed_results = []
        
        for condition in condition_columns:
            model = models.get(condition)
            if not model:
                logging.warning(f"النموذج لـ {condition} غير موجود")
                continue
            
            threshold = getattr(model, 'optimal_threshold', 0.5)
            proba = model.predict_proba(features_df)[0]
            positive_proba = float(proba[1])
            negative_proba = float(proba[0])
            prediction = 1 if positive_proba >= threshold else 0
            
            condition_name = NUMBER_TO_DISEASE.get(condition_columns.index(condition), condition)
            detailed_results.append({
                'condition': condition_name,
                'prediction': prediction,
                'confidence': positive_proba,
                'threshold': threshold
            })
            
            if prediction == 1:
                predictions[condition] = condition_name
                confidences[condition] = positive_proba
            
            all_probabilities[condition_name] = {
                'positive': positive_proba,
                'negative': negative_proba
            }
        
        if not predictions:
            return "لا توجد حالات صحية مؤكدة", 0.0, all_probabilities, detailed_results
        
        max_confidence = max(confidences.values())
        diagnosis = max(confidences, key=confidences.get)
        diagnosis_name = predictions[diagnosis]
        
        logging.info(f"التشخيص: {diagnosis_name} (ثقة: {max_confidence:.2%})")
        return diagnosis_name, max_confidence, all_probabilities, detailed_results
        
    except Exception as e:
        logging.error(f"خطأ في التنبؤ: {str(e)}", exc_info=True)
        raise RuntimeError("حدث خطأ أثناء التنبؤ") from e

# الحصول على شرح مفصل للمرض
def get_disease_explanation(disease_name, language='ar'):
    if disease_name == 'مرض الانسداد الرئوي المزمن':
        explanation = """
**مرض الانسداد الرئوي المزمن (COPD)**

**ما هو المرض؟**  
مرض الانسداد الرئوي المزمن هو حالة مزمنة تصيب الرئتين، مما يجعل التنفس صعبًا. يحدث بسبب تضرر الحويصلات الهوائية أو تضيق القنوات التي تحمل الهواء إلى الرئتين، مما يقلل من تدفق الهواء.

**الأعراض الشائعة:**  
- ضيق تنفس، خاصة أثناء النشاط البدني.  
- سعال مستمر، قد يكون مصحوبًا ببلغم.  
- الشعور بالتعب والإرهاق.  
- أصوات صفير أو أزيز أثناء التنفس.  
- تكرار الإصابة بالزكام أو التهابات الرئة.  
- فقدان الوزن أو تورم القدمين في الحالات المتقدمة.

**الأسباب وعوامل الخطر:**  
- **الأسباب الرئيسية:** التدخين، التعرض للغبار أو الغازات السامة، أو التدخين السلبي.  
- **عوامل الخطر:** التقدم في العمر، وجود تاريخ عائلي للمرض، أو الإصابة بأمراض رئوية مثل الربو.

**كيف يتم التشخيص؟**  
- فحص الطبيب للرئتين باستخدام سماعة طبية.  
- اختبار وظائف الرئة (سبيروميتر) لقياس تدفق الهواء.  
- تحاليل الدم لفحص مستويات الأكسجين.  
- أشعة الصدر (سينية أو مقطعية) لتقييم الرئتين.

**خيارات العلاج:**  
- **الأدوية:** أدوية لتوسيع القنوات الهوائية، أو الستيرويدات لتقليل الالتهاب، أو المضادات الحيوية للعدوى.  
- **الأكسجين الإضافي:** للحالات الشديدة.  
- **إعادة التأهيل الرئوي:** تمارين لتحسين التنفس.  
- **الجراحة:** في حالات نادرة وشديدة.

**نصائح للتعايش:**  
- الإقلاع عن التدخين فورًا.  
- ممارسة تمارين خفيفة مثل المشي حسب القدرة.  
- تناول نظام غذائي صحي ومتوازن.  
- تجنب التعرض للملوثات مثل الدخان أو الغبار.

**المضاعفات المحتملة:**  
- التهابات رئوية متكررة.  
- مشاكل في القلب.  
- ضعف عضلي أو فقدان الوزن.

**الوقاية:**  
- الابتعاد عن التدخين والملوثات.  
- تلقي لقاحات الإنفلونزا والالتهاب الرئوي.  
- إجراء فحوصات دورية إذا كنت في مجموعة الخطر.
"""
        return explanation.strip()
    
    prompt = f"""
    المريض يسأل عن مرض {disease_name}.
    يرجى شرح المرض باللغة {'العربية' if language == 'ar' else 'الإنجليزية'} بشكل واضح ومفصل يشمل:
    - تعريف مبسط للمرض
    - الأعراض الرئيسية والثانوية
    - الأسباب وعوامل الخطر
    - طرق التشخيص الطبية
    - خيارات العلاج المتاحة
    - نصائح للتعايش مع المرض
    - مضاعفات محتملة
    - طرق الوقاية (إن وجدت)
    
    أجب بلغة {'العربية' if language == 'ar' else 'الإنجليزية'} واضحة وسهلة الفهم، مع تجنب المصطلحات الطبية المعقدة.
    """
    return query_openrouter(prompt)

# تنسيق الرد
def format_response(diagnosis, symptoms, confidence, language='ar', include_explanation=False):
    disease_info = {
        'ar_name': diagnosis,
        'recommendations': 'راجع الطبيب لتشخيص دقيق وعلاج مناسب'
    }
    
    response = f"**التشخيص المحتمل:** {diagnosis}\n"
    response += f"**مستوى الثقة:** {confidence:.0%}\n\n"
    
    if include_explanation:
        detailed_explanation = get_disease_explanation(disease_name=diagnosis, language=language)
        response += "**📚 معلومات عن المرض:**\n"
        response += detailed_explanation + "\n\n"
    
    response += "**الأعراض التي ذكرتها:**\n"
    
    arabic_feature_names = {
        'GeneralHealth': 'الصحة العامة', 'PhysicalHealthDays': 'أيام الصحة البدنية',
        'MentalHealthDays': 'أيام الصحة العقلية', 'PhysicalActivities': 'الأنشطة البدنية',
        'SleepHours': 'ساعات النوم', 'DeafOrHardOfHearing': 'صعوبة السمع',
        'BlindOrVisionDifficulty': 'صعوبة الرؤية', 'DifficultyConcentrating': 'صعوبة التركيز',
        'DifficultyWalking': 'صعوبة المشي', 'DifficultyDressingBathing': 'صعوبة اللباس/الاستحمام',
        'DifficultyErrands': 'صعوبة القيام بالمهام', 'SmokerStatus': 'حالة التدخين',
        'ECigaretteUsage': 'استخدام السجائر الإلكترونية', 'AlcoholDrinkers': 'شرب الكحول',
        'HeightInMeters': 'الطول (متر)', 'WeightInKilograms': 'الوزن (كجم)', 'BMI': 'مؤشر كتلة الجسم',
        'Sex': 'الجنس', 'AgeCategory': 'فئة العمر'
    }
    
    for symptom, value in symptoms.items():
        if value not in [None, 0] and symptom in arabic_feature_names:
            response += f"- {arabic_feature_names[symptom]}: {value}\n"
    
    response += f"\n**🩺 التوصيات:** {disease_info['recommendations']}\n"
    response += "**⚠️ ملاحظة:** هذا التشخيص أولي ولا يغني عن استشارة الطبيب المختص."
    
    return response

# MAIN CHAT HANDLER
def handle_chat(user_input, language='ar'):
    """Main function to handle all user inputs"""
    try:
        # First check if it's a general query
        response = handle_general_query(user_input, language)
        if response:  # If template response exists
            return {
                'response': response,
                'needs_follow_up': False,
                'next_question': None
            }
        
        # Otherwise proceed with symptom analysis
        symptoms = extract_symptoms(user_input, feature_names, FEATURE_MAPPING)
        features = prepare_features(symptoms)
        
        # Check if we have enough info for diagnosis
        if all(v is not None for v in features.values()):
            diagnosis, confidence, _, _ = make_prediction(features)
            explanation = get_disease_explanation(diagnosis, language)
            return {
                'response': format_response(diagnosis, symptoms, confidence, language, True),
                'needs_follow_up': False,
                'next_question': None
            }
        else:
            next_q = determine_next_question({'symptoms': symptoms})
            if next_q:
                return {
                    'response': FOLLOW_UP_QUESTIONS[next_q]['text'],
                    'needs_follow_up': True,
                    'next_question': next_q
                }
            else:
                return {
                    'response': "لم أتمكن من فهم استفسارك. هل يمكنك إعادة صياغته؟",
                    'needs_follow_up': False,
                    'next_question': None
                }
                
    except Exception as e:
        logging.error(f"Error in chat handling: {str(e)}")
        return {
            'response': "حدث خطأ تقني. يرجى المحاولة مرة أخرى.",
            'needs_follow_up': False,
            'next_question': None
        }