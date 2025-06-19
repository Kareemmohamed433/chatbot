from flask import Flask, render_template, request, jsonify
import logging
import sys
import pandas as pd
from datetime import datetime
import uuid
from arabic_nlp import extract_symptoms, preprocess_text
from model import load_model_components, NUMBER_TO_DISEASE
from rl_agent import RLAgent

# إعداد التسجيل
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# تهيئة تطبيق Flask
app = Flask(__name__)
conversation_state = {}

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

# إنشاء تعيين الميزات
def generate_feature_mapping(features):
    """إنشاء قاموس لتحويل القيم النصية إلى قيم رقمية"""
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
            elif feature in ['HeightInMeters', 'WeightInKilograms']:
                mapping[feature] = lambda x: float(x) if x.replace('.', '', 1).isdigit() else (
                    1.70 if feature == 'HeightInMeters' else 1.0
                )
            elif feature == 'BMI':
                mapping[feature] = lambda x: max(10, min(50, float(x))) if x.replace('.', '', 1).isdigit() else 0
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
    """إنشاء قاموس الأسئلة المتابعة"""
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

# معالجة إجابة المستخدم
def process_answer(question, answer, answer_type):
    """معالجة إجابة المستخدم بناءً على نوع السؤال"""
    logging.debug(f"معالجة الإجابة لـ {question}: {answer} (النوع: {answer_type})")
    if answer_type == 'number':
        try:
            num = float(answer)
            return num if num >= 0 else 0
        except ValueError:
            return 0
    elif answer_type in ['gender', 'binary', 'category']:
        return FEATURE_MAPPING.get(question, {}).get(answer, 0)
    return 0

# تحضير الميزات
def prepare_features(symptoms):
    """تحضير الميزات للتنبؤ"""
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
                    FEATURE_MAPPING[feat]("") if feat in ['HeightInMeters', 'WeightInKilograms']
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
    """اختيار السؤال التالي بناءً على وكيل التعلم المعزز"""
    current_state = rl_agent.get_state(state['symptoms'])
    available_questions = [q for q in feature_names if state['symptoms'].get(q) is None]
    if not available_questions:
        logging.debug("لا توجد أسئلة متاحة متبقية")
        return None
    next_question = rl_agent.choose_action(current_state, available_questions)
    logging.debug(f"تم اختيار السؤال التالي: {next_question}")
    return next_question

# إجراء التنبؤ
def make_prediction(features):
    """إجراء التنبؤ باستخدام النماذج المدربة"""
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

# معلومات الأمراض
DISEASE_INFO = {
    'نوبة قلبية': {'ar_name': 'نوبة قلبية', 'recommendations': 'راجع طبيب قلب فورًا، تجنب الإجهاد'},
    'الذبحة الصدرية': {'ar_name': 'الذبحة الصدرية', 'recommendations': 'راجع طبيب قلب، تناول الأدوية الموصوفة'},
    'السكتة الدماغية': {'ar_name': 'السكتة الدماغية', 'recommendations': 'راجع طبيب أعصاب فورًا'},
    'الربو': {'ar_name': 'الربو', 'recommendations': 'استخدم البخاخ، تجنب المثيرات'},
    'سرطان الجلد': {'ar_name': 'سرطان الجلد', 'recommendations': 'راجع طبيب جلدية، إجراء فحوصات دورية'},
    'مرض الانسداد الرئوي المزمن': {'ar_name': 'مرض الانسداد الرئوي المزمن', 'recommendations': 'راجع طبيب رئة، توقف عن التدخين'},
    'الاكتئاب': {'ar_name': 'الاكتئاب', 'recommendations': 'راجع طبيب نفسي، جرب العلاج السلوكي'},
    'مرض الكلى': {'ar_name': 'مرض الكلى', 'recommendations': 'راجع طبيب كلى، اتبع نظامًا غذائيًا'},
    'التهاب المفاصل': {'ar_name': 'التهاب المفاصل', 'recommendations': 'راجع طبيب روماتيزم، مارس تمارين خفيفة'},
    'السكري': {'ar_name': 'السكري', 'recommendations': 'راجع طبيب غدد صماء، اتبع نظامًا غذائيًا'},
    'كوفيد-19': {'ar_name': 'كوفيد-19', 'recommendations': 'اعزل نفسك، راجع الطبيب إذا تفاقمت الأعراض'}
}

# تنسيق الرد
def format_response(diagnosis, symptoms, confidence):
    """تنسيق استجابة التشخيص للمستخدم"""
    disease_info = DISEASE_INFO.get(diagnosis, {
        'ar_name': diagnosis,
        'recommendations': 'راجع الطبيب لتشخيص دقيق'
    })
    response = f"التشخيص المحتمل: {disease_info['ar_name']}\n"
    response += f"مستوى الثقة: {confidence:.0%}\n"
    response += "الأعراض:\n"
    arabic_feature_names = {
        'GeneralHealth': 'الصحة العامة', 'PhysicalHealthDays': 'أيام الصحة البدنية',
        'MentalHealthDays': 'أيام الصحة العقلية', 'PhysicalActivities': 'الأنشطة البدنية',
        'SleepHours': 'ساعات النوم', 'DeafOrHardOfHearing': 'صعوبة السمع',
        'BlindOrVisionDifficulty': 'صعوبة الرؤية', 'DifficultyConcentrating': 'صعوبة التركيز',
        'DifficultyWalking': 'صعوبة المشي', 'DifficultyDressingBathing': 'صعوبة اللباس/الاستحمام',
        'DifficultyErrands': 'صعوبة القيام بالمهام', 'SmokerStatus': 'حالة التدخين',
        'ECigaretteUsage': 'استخدام السجائر الإلكترونية', 'AlcoholDrinkers': 'شرب الكحول'
    }
    for symptom, value in symptoms.items():
        if value not in [None, 0] and symptom in arabic_feature_names:
            response += f"- {arabic_feature_names[symptom]}: {value}\n"
    response += f"\nالتوصيات: {disease_info['recommendations']}"
    return response

# مسارات Flask
@app.route('/')
def index():
    """عرض الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """نقطة نهاية API للتشخيص الطبي"""
    try:
        if not request.is_json:
            return jsonify({'error': 'يجب أن يكون الطلب بصيغة JSON', 'status': 400}), 400
            
        data = request.get_json()
        user_input = data.get('message', '').strip()
        user_id = data.get('user_id', str(uuid.uuid4()))
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_input:
            return jsonify({'error': 'يرجى إدخال وصف للأعراض', 'status': 400}), 400
        
        if user_id not in conversation_state:
            conversation_state[user_id] = {
                'session_id': session_id,
                'stage': 'initial',
                'awaiting_answer': None,
                'symptoms': {feat: None for feat in feature_names},
                'responses': [],
                'start_time': datetime.now(),
                'last_interaction': datetime.now(),
                'asked_questions': set(),
                'diagnosis_history': []
            }
            logging.info(f"بدأت محادثة جديدة للمستخدم {user_id} (الجلسة: {session_id})")
        
        state = conversation_state[user_id]
        state['last_interaction'] = datetime.now()
        state['responses'].append(user_input)
        
        if state['awaiting_answer']:
            question = state['awaiting_answer']
            question_data = FOLLOW_UP_QUESTIONS.get(question)
            
            if not question_data:
                return jsonify({'error': 'خطأ في نظام الأسئلة', 'status': 500}), 500
                
            processed_value = process_answer(question, user_input, question_data['type'])
            
            if processed_value is None:
                return jsonify({
                    'error': 'إجابة غير صالحة',
                    'question': question_data['text'],
                    'options': question_data.get('options', []),
                    'question_type': question_data['type'],
                    'state': 'awaiting_response',
                    'status': 400
                }), 400
                
            state['symptoms'][question] = processed_value
            state['awaiting_answer'] = None
            state['asked_questions'].add(question)
            logging.info(f"تم تسجيل إجابة لـ {question}: {processed_value}")
        else:
            symptoms = extract_symptoms(user_input, feature_names, FEATURE_MAPPING)
            for key, value in symptoms.items():
                if value is not None and key in feature_names and state['symptoms'][key] is None:
                    state['symptoms'][key] = value
            
        missing_features = [feat for feat in feature_names if state['symptoms'][feat] is None]
        
        if not missing_features:
            features = prepare_features(state['symptoms'])
            diagnosis, confidence, all_probabilities, detailed_results = make_prediction(features)
            
            last_question = state['awaiting_answer'] or next(iter(feature_names), '')
            rl_agent.train(state['symptoms'], diagnosis, diagnosis, last_question)
            rl_agent.save()
            
            response = format_response(diagnosis, state['symptoms'], confidence)
            state['diagnosis_history'].append({
                'diagnosis': diagnosis,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            result = {
                'diagnosis': diagnosis,
                'confidence': float(confidence),
                'response': response,
                'state': 'complete',
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'probabilities': all_probabilities,
                'detailed_results': detailed_results,
                'status': 200
            }
            
            del conversation_state[user_id]
            return jsonify(result)
        
        next_question = determine_next_question(state)
        
        if not next_question:
            return jsonify({
                'error': 'لا يمكن تحديد الأسئلة الإضافية',
                'missing_features': missing_features,
                'state': 'error',
                'status': 400
            }), 400
            
        question_data = FOLLOW_UP_QUESTIONS.get(next_question)
        
        state['awaiting_answer'] = next_question
        state['stage'] = 'follow_up'
        
        return jsonify({
            'question': question_data['text'],
            'options': question_data.get('options', []),
            'question_type': question_data['type'],
            'state': 'awaiting_response',
            'progress': f"{len(feature_names) - len(missing_features)}/{len(feature_names)}",
            'session_id': session_id,
            'status': 200
        })
        
    except Exception as e:
        logging.error(f"خطأ في التشخيص: {str(e)}", exc_info=True)
        return jsonify({'error': 'حدث خطأ داخلي', 'status': 500}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """تنظيف حالة المحادثة"""
    try:
        global conversation_state
        conversation_state.clear()
        logging.info("تم مسح حالة المحادثة")
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logging.error(f"خطأ أثناء التنظيف: {str(e)}")
        return jsonify({'error': 'حدث خطأ أثناء التنظيف'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)