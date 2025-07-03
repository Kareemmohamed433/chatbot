# openrouter_response.py

import requests

# API Key for OpenRouter (replace with your actual key if needed)
OPENROUTER_API_KEY = "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"

def query_openrouter(prompt, model="mistralai/mistral-small-3.2-24b-instruct:free", temperature=0.5, max_tokens=500):
    """
    Query OpenRouter API with a given prompt and return the response.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-site-url.com",  # Recommended by OpenRouter
        "X-Title": "Medical Assistant"  # Recommended by OpenRouter
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "أنت مساعد طبي ذكي تشرح الأمراض للمستخدم باللغة العربية بشكل واضح ودقيق."
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
        return f"❌ خطأ في الاتصال: {str(e)}"
    except Exception as e:
        return f"❌ خطأ غير متوقع: {str(e)}"


def get_medical_info(disease_name):
    """
    Helper function to generate a medical prompt about a specific disease.
    """
    prompt = f"""
    المريض تم تشخيصه بمرض {disease_name}.
    يرجى شرح المرض باللغة العربية يشمل:
    - تعريف المرض
    - الأعراض الرئيسية
    - الأسباب وعوامل الخطر
    - طرق التشخيص
    - العلاج الموصى به
    - نصائح للوقاية (إن وجدت)
    - مضاعفات محتملة
    """
    return query_openrouter(prompt)


# ✅ Main execution for testing
if __name__ == "__main__":
    print("🩺 المساعد الطبي الذكي")
    print("-----------------------")

    # قائمة كلمات الترحيب الشائعة
    greetings = ['اهلا', 'أهلاً', 'مرحبا', 'مرحباً', 'السلام عليكم', 'هاي', 'هلا', 'صباح الخير', 'مساء الخير', 'hi', 'hello']

    while True:
        disease = input("\nأدخل اسم المرض الذي تريد معلومات عنه (أو 'خروج' للإنهاء): ").strip()

        if disease.lower() in ['خروج', 'exit', 'quit']:
            print("شكراً لاستخدامك المساعد الطبي. إلى اللقاء!")
            break

        if disease:
            if any(greet in disease.lower() for greet in greetings):
                print("\n🤖 أهلاً وسهلاً! أنا مساعدك الطبي الذكي. فقط أدخل اسم المرض الذي تريد معرفة معلومات عنه وسأساعدك.")
            else:
                print("\n🔍 جاري البحث عن المعلومات...")
                response = get_medical_info(disease)
                print("\n📘 معلومات عن المرض:\n")
                print(response)
                print("\n" + "="*50 + "\n")
        else:
            print("⚠️ الرجاء إدخال اسم مرض صحيح.")
