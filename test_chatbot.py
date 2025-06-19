import requests
import json
import uuid
import time

# تعريف عنوان API الأساسي
BASE_URL = "http://0.0.0.0:5000/api/diagnose"
CLEANUP_URL = "http://0.0.0.0:5000/api/cleanup"
USER_ID = str(uuid.uuid4())
SESSION_ID = str(uuid.uuid4())

def send_message(message):
    """إرسال رسالة إلى الشاتبوت وعرض الرد"""
    payload = {
        "message": message,
        "user_id": USER_ID,
        "session_id": SESSION_ID
    }
    try:
        response = requests.post(BASE_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"\n>>> أرسلت: {message}")
        print("<<< الرد:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.json()
    except requests.RequestException as e:
        print(f"خطأ في الاتصال: {e}")
        return None

def cleanup_conversation():
    """تنظيف حالة المحادثة عبر نقطة نهاية /api/cleanup"""
    try:
        response = requests.post(CLEANUP_URL, timeout=10)
        response.raise_for_status()
        print("\n>>> طلب تنظيف المحادثة")
        print(f"<<< رد التنظيف: {response.json()}")
    except requests.RequestException as e:
        print(f"خطأ في تنظيف المحادثة: {e}")

def run_tests():
    """تشغيل اختبارات الشاتبوت"""
    print("=== بدء اختبار الشاتبوت ===")
    
    # اختبار 1: تحية
    print("\n=== اختبار 1: إرسال تحية ('أهلان') ===")
    response = send_message("أهلان")
    time.sleep(1)
    
    # اختبار 2: إدخال غامض
    print("\n=== اختبار 2: إدخال غامض ('تعبان وكده') ===")
    response = send_message("تعبان وكده")
    time.sleep(1)
    
    # اختبار 3: الرد على سؤال بداية الأعراض
    if response and 'question' in response and 'بداية الأعراض' in response['question']:
        print("\n=== اختبار 3: الرد على سؤال بداية الأعراض ('منذ ساعة') ===")
        send_message("منذ ساعة")
        time.sleep(1)
    
    # اختبار 4: إدخال أعراض محددة
    print("\n=== اختبار 4: إدخال أعراض محددة ('ألم في الصدر وضيق تنفس') ===")
    send_message("ألم في الصدر وضيق تنفس")
    time.sleep(1)
    
    # اختبار 5: إدخال وزن غير منطقي
    print("\n=== اختبار 5: إدخال وزن غير منطقي ('1') ===")
    response = send_message("وزني 1 كجم")  # يفترض أن يتم التعرف على الوزن
    time.sleep(1)
    
    # اختبار 6: إدخال فارغ
    print("\n=== اختبار 6: إدخال فارغ ===")
    send_message("")
    time.sleep(1)
    
    # اختبار 7: تنظيف المحادثة
    print("\n=== اختبار 7: تنظيف المحادثة ===")
    cleanup_conversation()

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\n=== تم إيقاف الاختبار ===")
        cleanup_conversation()