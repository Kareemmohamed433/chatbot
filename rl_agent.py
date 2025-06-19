import numpy as np
import pickle
import logging
import sys

# إعداد التسجيل
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class RLAgent:
    def __init__(self, questions, diseases):
        self.questions = questions
        self.diseases = diseases
        self.q_table = {}
        self.alpha = 0.1  # معدل التعلم
        self.gamma = 0.9  # عامل الخصم
        self.epsilon = 0.1  # معدل الاستكشاف
        self.asked_questions = set()

    def get_state(self, symptoms):
        """إنشاء حالة بناءً على الأعراض الحالية"""
        state = [(key, symptoms.get(key)) for key in self.questions]
        return tuple(state)

    def initialize_q_table(self, state, actions):
        """تهيئة جدول Q إذا لم يكن موجودًا"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in actions}

    def choose_action(self, state, available_actions):
        """اختيار السؤال التالي باستخدام استراتيجية epsilon-greedy"""
        self.initialize_q_table(state, available_actions)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(available_actions)
            logging.debug(f"استكشاف: تم اختيار السؤال {action}")
        else:
            action_values = {action: self.q_table[state][action] for action in available_actions}
            action = max(action_values, key=action_values.get)
            logging.debug(f"استغلال: تم اختيار السؤال {action}")
        self.asked_questions.add(action)
        return action

    def calculate_reward(self, predicted_disease, true_disease):
        """حساب المكافأة بناءً على دقة التنبؤ"""
        return 1.0 if predicted_disease == true_disease else -1.0

    def train(self, symptoms, predicted_disease, true_disease, question):
        """تحديث جدول Q بناءً على المكافأة"""
        state = self.get_state(symptoms)
        reward = self.calculate_reward(predicted_disease, true_disease)
        self.initialize_q_table(state, self.questions)
        next_state = state
        self.initialize_q_table(next_state, self.questions)
        current_q = self.q_table[state].get(question, 0.0)
        max_future_q = max(self.q_table[next_state].values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][question] = new_q

    def save(self):
        """حفظ جدول Q في ملف"""
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
        logging.info("تم حفظ جدول Q بنجاح")

    def load(self):
        """تحميل جدول Q من ملف"""
        try:
            with open('q_table.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
            logging.info("تم تحميل جدول Q بنجاح")
        except FileNotFoundError:
            logging.warning("ملف جدول Q غير موجود، يتم إنشاء جدول جديد")
            self.q_table = {}