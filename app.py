import uuid
import os
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
from arabic_nlp import (
    extract_disease_name, extract_symptoms, handle_general_query,
    process_answer, prepare_features, determine_next_question,
    make_prediction, format_response, get_disease_explanation,
    FEATURE_MAPPING, FOLLOW_UP_QUESTIONS, rl_agent, feature_names
)
import logging

# إعداد التسجيل مع ترميز UTF-8
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# تهيئة تطبيق Flask
app = Flask(__name__)
conversation_state = {}

# مسارات Flask
@app.route('/')
def index():
    try:
        logging.info("Serving index.html")
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error serving index.html: {str(e)}")
        return jsonify({'error': 'خطأ في تحميل الصفحة الرئيسية'}), 500

@app.route('/api/start_chat', methods=['POST'])
def start_chat():
    try:
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        conversation_state[(user_id, session_id)] = {
            'stage': 'greeting',
            'symptoms': {feat: None for feat in feature_names},
            'responses': [],
            'asked_questions': set(),
            'awaiting_answer': None,
            'diagnosis_history': [],
            'last_interaction': datetime.now(),
            'chat_history': [],
            'last_diagnosis': None
        }
        welcome_message = handle_general_query("مرحبا", language='ar')
        return jsonify({
            'status': 200,
            'message': welcome_message,
            'user_id': user_id,
            'session_id': session_id
        })
    except Exception as e:
        logging.error(f"Error in start_chat: {str(e)}", exc_info=True)
        return jsonify({'error': 'حدث خطأ أثناء بدء المحادثة', 'status': 500}), 500

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    try:
        logging.info(f"Incoming request data: {request.get_data(as_text=True)}")
        
        if not request.is_json:
            return jsonify({'error': 'يجب أن يكون الطلب بصيغة JSON', 'status': 400}), 400
            
        data = request.get_json()
        user_input = data.get('message', '').strip()
        user_id = data.get('user_id', '')
        session_id = data.get('session_id', '')
        language = data.get('language', 'ar')
        explain_disease = data.get('explain_disease', False)

        if not user_id or not session_id or (user_id, session_id) not in conversation_state:
            return jsonify({
                'error': 'لم يتم التعرف على المستخدم، يرجى بدء محادثة جديدة',
                'status': 400,
                'action': 'start_new_chat'
            }), 400

        if not user_input:
            next_question = determine_next_question(conversation_state[(user_id, session_id)])
            if next_question:
                question_data = FOLLOW_UP_QUESTIONS.get(next_question)
                conversation_state[(user_id, session_id)]['awaiting_answer'] = next_question
                return jsonify({
                    'question': question_data['text'],
                    'options': question_data.get('options', []),
                    'question_type': question_data['type'],
                    'state': 'awaiting_response',
                    'progress': f"{len([x for x in conversation_state[(user_id, session_id)]['symptoms'].values() if x is not None])}/{len(feature_names)}",
                    'session_id': session_id,
                    'status': 200
                })
            return jsonify({'error': 'يرجى إدخال وصف للأعراض أو استفسار طبي', 'status': 400}), 400

        state = conversation_state[(user_id, session_id)]
        state['last_interaction'] = datetime.now()
        state['responses'].append(user_input)
        state['chat_history'].append({'role': 'user', 'content': user_input})

        general_keywords = ['ما هو', 'شرح', 'معلومات عن', 'كيف', 'أهلان', 'اهلا', 'مرحبا', 'اشرح المرض']
        is_general_query = any(keyword in user_input.lower() for keyword in general_keywords)

        if is_general_query:
            logging.info(f"تم اكتشاف استفسار عام: {user_input}")

            if 'اشرح المرض' in user_input.lower():
                disease_name = extract_disease_name(user_input)

                if disease_name:
                    response = get_disease_explanation(disease_name, language)
                elif state['last_diagnosis']:
                    response = get_disease_explanation(state['last_diagnosis'], language)
                else:
                    response = "لم يتم تشخيص أي مرض بعد. يرجى وصف أعراضك أولاً."

                state['chat_history'].append({'role': 'assistant', 'content': response})
                return jsonify({
                    'message': response,
                    'state': 'complete',
                    'session_id': session_id,
                    'status': 200
                })

            elif 'ما هو' in user_input.lower() and 'المرض' in user_input.lower() and state['diagnosis_history']:
                last_diagnosis = state['diagnosis_history'][-1]['diagnosis']
                response = get_disease_explanation(last_diagnosis, language)
                state['chat_history'].append({'role': 'assistant', 'content': response})
                return jsonify({
                    'message': response,
                    'state': 'complete',
                    'session_id': session_id,
                    'status': 200
                })

            else:
                response = handle_general_query(user_input, language)
                state['chat_history'].append({'role': 'assistant', 'content': response})
                return jsonify({
                    'message': response,
                    'state': 'complete',
                    'session_id': session_id,
                    'status': 200
                })

        elif state['awaiting_answer']:
            question = state['awaiting_answer']
            question_data = FOLLOW_UP_QUESTIONS.get(question)

            if not question_data:
                return jsonify({'error': 'خطأ في نظام الأسئلة', 'status': 500}), 500

            processed_value, error_message = process_answer(question, user_input, question_data['type'])

            if processed_value is None:
                return jsonify({
                    'error': error_message,
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

        missing_features = [feat for feat in feature_names if state['symptoms'][feat] is None and feat != 'BMI']

        if missing_features and state['stage'] != 'diagnosis':
            next_question = determine_next_question(state)

            if next_question:
                question_data = FOLLOW_UP_QUESTIONS.get(next_question)
                state['awaiting_answer'] = next_question

                return jsonify({
                    'question': question_data['text'],
                    'options': question_data.get('options', []),
                    'question_type': question_data['type'],
                    'state': 'awaiting_response',
                    'progress': f"{len([x for x in state['symptoms'].values() if x is not None])}/{len(feature_names)}",
                    'session_id': session_id,
                    'status': 200
                })

        features = prepare_features(state['symptoms'])
        diagnosis, confidence, all_probabilities, detailed_results = make_prediction(features)

        last_question = state['awaiting_answer'] or next(iter(feature_names), '')
        rl_agent.train(state['symptoms'], diagnosis, diagnosis, last_question)
        rl_agent.save()

        include_explanation = explain_disease or 'اشرح المرض' in user_input.lower()
        response = format_response(diagnosis, state['symptoms'], confidence, language, include_explanation)

        state['diagnosis_history'].append({
            'diagnosis': diagnosis,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'detailed_probabilities': all_probabilities
        })
        state['last_diagnosis'] = diagnosis

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

        return jsonify(result)

    except Exception as e:
        logging.error(f"خطأ في التشخيص: {str(e)}", exc_info=True)
        return jsonify({'error': 'حدث خطأ داخلي', 'status': 500}), 500

@app.route('/api/voice-to-text', methods=['POST'])
def voice_to_text():
    try:
        if 'audio' not in request.files:
            logging.error("لم يتم استلام ملف صوتي في الطلب")
            return jsonify({'error': 'لم يتم استلام ملف صوتي', 'status': 400}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            logging.error("ملف صوتي فارغ تم استلامه")
            return jsonify({'error': 'ملف صوتي فارغ', 'status': 400}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            original_path = tmp_file.name
            audio_file.save(original_path)

        try:
            audio = AudioSegment.from_file(original_path)
            wav_path = original_path.rsplit('.', 1)[0] + '_converted.wav'
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio.export(wav_path, format='wav')
        except Exception as e:
            logging.error(f"خطأ في تحويل الملف الصوتي باستخدام pydub: {str(e)}")
            os.remove(original_path)
            return jsonify({
                'error': 'خطأ في معالجة الملف الصوتي',
                'status': 500,
                'details': str(e)
            }), 500

        try:
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data, language='ar-SA')
            logging.info(f"تم التعرف على الصوت: {text}")
            return jsonify({
                'text': text,
                'status': 200,
                'language': 'ar',
                'confidence': 1.0
            })
        except sr.UnknownValueError:
            logging.warning("لم يتم فهم الصوت من قبل خدمة التعرف")
            return jsonify({
                'error': 'لم يتم فهم الصوت',
                'status': 400,
                'suggestions': ['حاول التحدث بوضوح أكثر', 'تحدث بصوت أعلى']
            }), 400
        except sr.RequestError as e:
            logging.error(f"خطأ في خدمة التعرف على الصوت: {str(e)}")
            return jsonify({
                'error': 'مشكلة في خدمة التعرف على الصوت',
                'status': 503,
                'alternative': 'يرجى المحاولة لاحقًا أو استخدام الإدخال النصي'
            }), 503
        finally:
            for path in [original_path, wav_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logging.warning(f"فشل في حذف الملف المؤقت {path}: {str(e)}")

    except Exception as e:
        logging.error(f"خطأ فادح في التعرف على الصوت: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'حدث خطأ غير متوقع',
            'status': 500,
            'details': str(e)
        }), 500

@app.route('/api/text-to-voice', methods=['POST'])
def text_to_voice():
    try:
        data = request.get_json()
        response_text = data.get('text', '')
        language = data.get('language', 'ar')
        speed = data.get('speed', 150)

        if not response_text:
            return jsonify({'error': 'النص فارغ', 'status': 400}), 400

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        arabic_voice = None

        for voice in voices:
            if hasattr(voice, 'languages') and 'ar' in str(voice.languages).lower():
                arabic_voice = voice.id
                break
            elif 'ar' in voice.id.lower():
                arabic_voice = voice.id
                break

        if arabic_voice:
            engine.setProperty('voice', arabic_voice)
        else:
            logging.warning("لم يتم العثور على صوت عربي، استخدام الصوت الافتراضي")

        engine.setProperty('rate', speed)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            temp_path = tmp_file.name

        engine.save_to_file(response_text, temp_path)
        engine.runAndWait()

        response = send_file(
            temp_path,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='response.mp3'
        )

        @response.call_on_close
        def cleanup():
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return response

    except Exception as e:
        logging.error(f"خطأ في تحويل النص إلى صوت: {str(e)}")
        return jsonify({
            'error': 'حدث خطأ في التشغيل الصوتي',
            'status': 500,
            'solution': 'يرجى التأكد من تثبيت جميع المكتبات الصوتية'
        }), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    try:
        current_time = datetime.now()
        expired = [
            key for key, state in conversation_state.items()
            if (current_time - state['last_interaction']).total_seconds() > 3600
        ]
        for key in expired:
            del conversation_state[key]
        logging.info("تم مسح حالات المحادثة القديمة")
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logging.error(f"خطأ أثناء التنظيف: {str(e)}")
        return jsonify({'error': 'حدث خطأ أثناء التنظيف'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)