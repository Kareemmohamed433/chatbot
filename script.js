let userId = null;
let sessionId = null;
let language = 'ar';
let isRecording = false;
let recognition = null;
let isSpeechEnabled = true;
let fabOpen = false;
let lastMessageTime = null;
let newMessageCount = 0;
let hasUserInteracted = false;
let expectedOptions = [];

// Detect mobile device
function isMobileDevice() {
    return (typeof window.orientation !== "undefined") || (navigator.userAgent.indexOf('IEMobile') !== -1);
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    await startChat();
    initializeSpeechRecognition();
    setupEventListeners();
    initializeFAB();
    initializeAutoResize();
    initializeConfetti();
    initializeMobileFeatures();

    // Detect user interaction
    ['click', 'touchstart'].forEach(event => {
        document.addEventListener(event, () => {
            hasUserInteracted = true;
        }, { once: true });
    });
});

// Initialize mobile-specific features
function initializeMobileFeatures() {
    if (isMobileDevice()) {
        // Add touch effects to buttons
        document.querySelectorAll('button').forEach(button => {
            button.addEventListener('touchstart', () => {
                button.style.transform = 'scale(0.95)';
            });
            button.addEventListener('touchend', () => {
                button.style.transform = '';
            });
        });
        
        // Prevent zooming on input focus
        document.getElementById('user-input').addEventListener('focus', () => {
            document.body.style.zoom = "1.0";
            setTimeout(() => {
                window.scrollTo(0, 0);
                document.body.scrollTop = 0;
            }, 100);
        });
        
        // Adjust viewport for mobile
        const viewportMeta = document.querySelector('meta[name="viewport"]');
        if (viewportMeta) {
            viewportMeta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
        }
    }
}

// Set up all event listeners
function setupEventListeners() {
    document.getElementById('send-button').addEventListener('click', () => sendMessage());
    document.getElementById('user-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    document.getElementById('attach-button').addEventListener('click', () => {
        showModal(language === 'en' ? 'File attachment feature coming soon!' : 'ميزة إرفاق الملف قريبًا!');
    });
    document.getElementById('main-fab').addEventListener('click', toggleFAB);
    document.getElementById('voice-fab').addEventListener('click', toggleVoiceRecording);
    document.getElementById('language-fab').addEventListener('click', toggleLanguage);
    document.getElementById('restart-fab').addEventListener('click', restartChat);
    document.querySelectorAll('.quick-suggestion').forEach(button => {
        button.addEventListener('click', () => sendMessage(button.textContent));
    });
    
    // Scroll to bottom when new messages arrive
    const chatContainer = document.getElementById('chat-container');
    const observer = new MutationObserver(() => {
        if (isScrolledToBottom()) {
            adjustLayout();
        } else {
            showNewMessageIndicator();
        }
    });
    observer.observe(chatContainer, { childList: true });
}

// Initialize floating action button
function initializeFAB() {
    document.getElementById('main-fab').addEventListener('click', toggleFAB);
}

function toggleFAB() {
    const fabContainer = document.querySelector('.fab-container');
    fabOpen = !fabOpen;
    fabContainer.classList.toggle('open', fabOpen);
}

// Initialize textarea auto-resize
function initializeAutoResize() {
    const textarea = document.getElementById('user-input');
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

// Initialize confetti effect
function initializeConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

// Check if chat is scrolled to bottom
function isScrolledToBottom() {
    const chatContainer = document.getElementById('chat-container');
    return chatContainer.scrollHeight - chatContainer.clientHeight <= chatContainer.scrollTop + 10;
}

// Show new message indicator
function showNewMessageIndicator() {
    const indicator = document.querySelector('.new-message-indicator');
    newMessageCount++;
    indicator.querySelector('span').textContent = language === 'en' ? 
        `${newMessageCount} new message${newMessageCount > 1 ? 's' : ''}` : 
        `${newMessageCount} رسالة جديدة`;
    indicator.classList.add('visible');
    
    // Scroll to bottom when indicator is clicked
    indicator.addEventListener('click', () => {
        adjustLayout();
        indicator.classList.remove('visible');
        newMessageCount = 0;
    });
}

// Start a new chat session
async function startChat() {
    try {
        showLoadingAnimation();
        const response = await fetch('/api/start_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        hideLoadingAnimation();
        
        if (data.status === 200) {
            userId = data.user_id;
            sessionId = data.session_id;
            addMessage(data.message, 'bot-message');
            if (hasUserInteracted) {
                await textToSpeech(data.message);
            }
            triggerConfetti();
        } else {
            showModal(data.error || 'حدث خطأ أثناء بدء المحادثة');
        }
    } catch (error) {
        hideLoadingAnimation();
        showModal('خطأ في الاتصال بالخادم، يرجى المحاولة لاحقًا');
        console.error('Error starting chat:', error);
    }
}

// Show loading animation
function showLoadingAnimation() {
    const chatContainer = document.getElementById('chat-container');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-animation';
    loadingDiv.innerHTML = `
        <div class="loading-spinner"></div>
        <p>${language === 'en' ? 'Starting chat...' : 'جاري بدء المحادثة...'}</p>
    `;
    chatContainer.appendChild(loadingDiv);
}

// Hide loading animation
function hideLoadingAnimation() {
    const loadingDiv = document.querySelector('.loading-animation');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// Trigger confetti effect
function triggerConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    const confetti = canvas.getContext('2d');
    const confettiPieces = [];
    const pieceCount = 150;
    
    for (let i = 0; i < pieceCount; i++) {
        confettiPieces.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height - canvas.height,
            r: Math.random() * 4 + 1,
            d: Math.random() * pieceCount,
            color: `hsl(${Math.random() * 360}, 100%, 50%)`,
            tilt: Math.floor(Math.random() * 10) - 10,
            tiltAngle: Math.random() * 0.1,
            tiltAngleIncrement: Math.random() * 0.07
        });
    }
    
    function drawConfetti() {
        confetti.clearRect(0, 0, canvas.width, canvas.height);
        
        let allLanded = true;
        
        confettiPieces.forEach((p) => {
            confetti.beginPath();
            confetti.lineWidth = p.r / 2;
            confetti.strokeStyle = p.color;
            confetti.moveTo(p.x + p.tilt, p.y);
            confetti.lineTo(p.x + p.tilt + p.r * 2, p.y);
            confetti.stroke();
            
            p.tiltAngle += p.tiltAngleIncrement;
            p.y += (Math.cos(p.d) + 3 + p.r / 2) / 2;
            p.tilt = Math.sin(p.tiltAngle) * 15;
            
            if (p.y <= canvas.height) {
                allLanded = false;
            }
        });
        
        if (!allLanded) {
            requestAnimationFrame(drawConfetti);
        } else {
            confetti.clearRect(0, 0, canvas.width, canvas.height);
        }
    }
    
    drawConfetti();
}

// Request explanation for the current diagnosis
function requestDiseaseExplanation() {
    const explainText = language === 'en' ? 'Explain this condition' : 'اشرح هذا المرض';
    sendMessage(explainText);
}

// Send a message to the chatbot
async function sendMessage(message = null) {
    const inputField = document.getElementById('user-input');
    const messageText = message || inputField.value.trim();
    
    if (!messageText) {
        showModal(language === 'en' ? 'Please enter a description of your symptoms or a question.' : 'يرجى إدخال وصف للأعراض أو سؤال.');
        return;
    }
    
    if (expectedOptions.length > 0 && !expectedOptions.includes(messageText)) {
        showModal(language === 'en' ? 
            `Please select one of the provided options: ${expectedOptions.join(', ')}` :
            `يرجى اختيار إجابة من الخيارات المتاحة: ${expectedOptions.join('، ')}`);
        inputField.value = '';
        return;
    }
    
    if (!userId || !sessionId) {
        showModal(language === 'en' ? 'Starting a new chat session...' : 'جاري بدء محادثة جديدة...');
        await startChat();
        return;
    }
    
    if (!message) {
        addMessage(messageText, 'user-message');
        inputField.value = '';
        inputField.style.height = 'auto';
        if (isMobileDevice()) {
            inputField.blur();
        }
        expectedOptions = [];
    }
    
    showTypingIndicator();
    
    try {
        const isExplainRequest = messageText.includes('اشرح المرض') || messageText.includes('Explain this condition');
        const requestBody = {
            message: messageText,
            user_id: userId,
            session_id: sessionId,
            language: language,
            explain_disease: isExplainRequest
        };

        const response = await fetch('/api/diagnose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }
        
        const data = await response.json();
        hideTypingIndicator();
        
        if (data.error) {
            addMessage(data.error, 'bot-message');
            if (data.action === 'start_new_chat') {
                showModal(language === 'en' ? 'Session expired. Starting a new chat...' : 'انتهت الجلسة. جاري بدء محادثة جديدة...');
                await startChat();
                return;
            }
            if (data.question) {
                addMessageWithOptions(data.question, data.options || [], data.question_type);
            }
        } else if (data.state === 'awaiting_response') {
            addMessageWithOptions(data.question, data.options || [], data.question_type);
        } else if (data.state === 'complete') {
            if (data.diagnosis) {
                displayDiagnosis(data);
                await textToSpeech(data.response);
            } else {
                addMessage(data.message || data.response, 'bot-message');
                await textToSpeech(data.message || data.response);
            }
        }
        
        adjustLayout();
    } catch (error) {
        hideTypingIndicator();
        showModal(language === 'en' ? 
            `Sorry, an error occurred: ${error.message}. Please try again.` : 
            `عذرًا، حدث خطأ: ${error.message}. يرجى المحاولة مرة أخرى.`);
        console.error('Error:', error);
    }
}

// Display diagnosis information
function displayDiagnosis(data) {
    const chatContainer = document.getElementById('chat-container');
    
    let severityClass = 'severity-medium';
    let severityText = language === 'en' ? 'Medium' : 'متوسطة';
    let diagnosisClass = 'diagnosis-card';
    let severityIcon = '<i class="fas fa-info-circle"></i>';
    
    if (data.confidence < 0.65) {
        severityClass = 'severity-high';
        severityText = language === 'en' ? 'High' : 'عالية';
        diagnosisClass = 'diagnosis-card high-risk';
        severityIcon = '<i class="fas fa-exclamation-triangle"></i>';
    } else if (data.confidence >= 0.8) {
        severityClass = 'severity-low';
        severityText = language === 'en' ? 'Low' : 'منخفضة';
        severityIcon = '<i class="fas fa-check-circle"></i>';
    }
    
    const diagnosisDiv = document.createElement('div');
    diagnosisDiv.className = diagnosisClass;
    
    diagnosisDiv.innerHTML = `
        <div class="diagnosis-title">
            <i class="fas fa-heartbeat"></i>
            <span>${language === 'en' ? 'Possible Condition' : 'الحالة المحتملة'}</span>
        </div>
        <div class="diagnosis-text">${data.diagnosis}</div>
        
        <div class="confidence-meter">
            <div class="confidence-level" style="width: ${data.confidence * 100}%"></div>
        </div>
        <div class="confidence-label">
            ${severityIcon}
            ${language === 'en' ? 'Confidence' : 'مستوى الثقة'}: ${Math.round(data.confidence * 100)}%
            <span class="${severityClass}">${severityText}</span>
        </div>
    `;
    
    if (data.confidence < 0.7 && data.diagnosis.toLowerCase().includes(language === 'en' ? 'heart' : 'قلب')) {
        const emergencyDiv = document.createElement('div');
        emergencyDiv.className = 'emergency-alert';
        emergencyDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <div class="emergency-text">
                ${language === 'en' ? 
                    'Your symptoms may indicate a serious condition. Please seek immediate medical attention.' : 
                    'قد تشير أعراضك إلى حالة خطيرة. يرجى التماس العناية الطبية الفورية.'}
                <div class="emergency-cta">
                    ${language === 'en' ? 
                        'Call emergency services now:' : 
                        'اتصل بخدمات الطوارئ الآن:'}
                    <a href="tel:911" class="emergency-call">
                        <i class="fas fa-phone"></i> 911
                    </a>
                </div>
            </div>
        `;
        diagnosisDiv.appendChild(emergencyDiv);
    }
    
    if (data.response) {
        const recDiv = document.createElement('div');
        recDiv.className = 'recommendations';
        
        const recTitle = document.createElement('div');
        recTitle.className = 'recommendations-title';
        recTitle.innerHTML = `<i class="fas fa-clipboard-list"></i>${language === 'en' ? 'Explanation and Recommendations' : 'التفسير والتوصيات'}`;
        
        const recText = document.createElement('div');
        recText.className = 'recommendations-text';
        recText.textContent = data.response;
        
        recDiv.appendChild(recTitle);
        recDiv.appendChild(recText);
        diagnosisDiv.appendChild(recDiv);
    }
    
    if (!data.response.includes('معلومات عن المرض') && !data.response.includes('Information about the disease')) {
        const explainButton = document.createElement('button');
        explainButton.className = 'option-btn explain-button';
        explainButton.textContent = language === 'en' ? 'Explain this condition' : 'اشرح هذا المرض';
        explainButton.onclick = requestDiseaseExplanation;
        diagnosisDiv.appendChild(explainButton);
    }
    
    chatContainer.appendChild(diagnosisDiv);
    
    setTimeout(() => {
        addMessage(
            language === 'en' ? 'Do you have other symptoms or questions?' : 'هل لديك أعراض أو أسئلة أخرى؟',
            'bot-message'
        );
    }, 1000);
}

// Add a message to the chat
function addMessage(text, className) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-content">${text}</div>
        <div class="message-time">${timeString}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    lastMessageTime = now;
    adjustLayout();
}

// Add message with options
function addMessageWithOptions(question, options, questionType) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    messageDiv.innerHTML = `
        <div class="message-content">${question}</div>
    `;
    
    if (options && options.length > 0) {
        expectedOptions = options;
        const optionsDiv = document.createElement('div');
        optionsDiv.className = 'options';
        
        options.forEach(option => {
            const button = document.createElement('button');
            button.className = 'option-btn';
            button.textContent = option;
            button.addEventListener('click', () => {
                expectedOptions = [];
                sendMessage(option);
            });
            optionsDiv.appendChild(button);
        });
        
        messageDiv.appendChild(optionsDiv);
    }
    
    chatContainer.appendChild(messageDiv);
    adjustLayout();
}

// Adjust chat layout
function adjustLayout() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    if (isMobileDevice()) {
        setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 100);
    }
    
    document.querySelector('.new-message-indicator').classList.remove('visible');
    newMessageCount = 0;
}

// Show typing indicator
function showTypingIndicator() {
    const chatContainer = document.getElementById('chat-container');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    chatContainer.appendChild(typingDiv);
    adjustLayout();
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.querySelector('.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Show modal dialog
function showModal(message) {
    const modal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    modal.classList.add('active');
}

// Close modal dialog
function closeModal() {
    document.getElementById('error-modal').classList.remove('active');
}

// Initialize speech recognition
function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = language === 'en' ? 'en-US' : 'ar-SA';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.continuous = false;
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            document.getElementById('voice-transcript').textContent = transcript;
            setTimeout(() => {
                addMessage(transcript, 'user-message');
                sendMessage(transcript);
            }, 500);
            toggleVoiceRecording();
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            if (event.error === 'no-speech') {
                showModal(language === 'en' ? 'No speech detected. Please try again.' : 'لم يتم الكشف عن صوت. يرجى المحاولة مرة أخرى.');
            } else {
                showModal(language === 'en' ? `Speech recognition error: ${event.error}` : `خطأ في التعرف على الصوت: ${event.error}`);
            }
            toggleVoiceRecording();
        };
        
        recognition.onend = () => {
            if (isRecording) {
                toggleVoiceRecording();
            }
        };
    } else {
        document.getElementById('voice-fab').style.display = 'none';
        console.warn('Speech recognition not supported in this browser');
    }
}

// Toggle voice recording
function toggleVoiceRecording() {
    const voiceModal = document.getElementById('voice-modal');
    const voiceInstruction = document.getElementById('voice-instruction');
    const voiceFab = document.getElementById('voice-fab');
    
    if (!isRecording) {
        isRecording = true;
        voiceModal.classList.add('active');
        voiceInstruction.textContent = language === 'en' ? 'Speak now...' : 'يتحدث الآن...';
        voiceFab.classList.add('recording');
        document.getElementById('voice-transcript').textContent = '';
        
        if (recognition) {
            recognition.lang = language === 'en' ? 'en-US' : 'ar-SA';
            try {
                recognition.start();
                setTimeout(() => {
                    if (isRecording) {
                        toggleVoiceRecording();
                        showModal(language === 'en' ? 'No speech detected. Recording stopped.' : 'لم يتم الكشف عن صوت. تم إيقاف التسجيل.');
                    }
                }, 10000);
            } catch (error) {
                console.error('Failed to start speech recognition:', error);
                isRecording = false;
                voiceModal.classList.remove('active');
                voiceFab.classList.remove('recording');
            }
        }
    } else {
        isRecording = false;
        voiceModal.classList.remove('active');
        voiceFab.classList.remove('recording');
        if (recognition) {
            try {
                recognition.stop();
            } catch (error) {
                console.error('Failed to stop speech recognition:', error);
            }
        }
    }
    
    toggleFAB();
}

// Convert text to speech
async function textToSpeech(text) {
    if (!isSpeechEnabled) return;
    
    try {
        const response = await fetch('/api/text-to-voice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, language, speed: 150 })
        });
        
        if (response.ok) {
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('audio')) {
                throw new Error('Invalid audio response from server');
            }
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            if (hasUserInteracted) {
                audio.play();
                audio.onended = () => URL.revokeObjectURL(audioUrl);
            } else {
                console.warn('Audio playback skipped: No user interaction');
                URL.revokeObjectURL(audioUrl);
            }
        } else {
            const errorData = await response.json();
            console.error('Failed to convert text to speech:', errorData.error || response.statusText);
            showModal(language === 'en' ? 'Failed to generate audio. Please try again.' : 'فشل في إنشاء الصوت. يرجى المحاولة مرة أخرى.');
        }
    } catch (error) {
        console.error('Error in text-to-speech:', error);
        showModal(language === 'en' ? `Text-to-speech error: ${error.message}` : `خطأ في تحويل النص إلى صوت: ${error.message}`);
    }
}

// Toggle language
function toggleLanguage() {
    language = language === 'ar' ? 'en' : 'ar';
    document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr';
    document.documentElement.lang = language;
    updateUIText();
    toggleFAB();
}

// Update UI text based on language
function updateUIText() {
    document.querySelector('.header-content h1').textContent = language === 'en' ? 'Smart Medical Assistant' : 'المساعد الطبي الذكي';
    document.querySelector('.header-content p').textContent = language === 'en' ? 
        'Enter your symptoms or ask a medical question, and I\'ll provide accurate information.' :
        'أدخل أعراضك أو اطرح استفسارًا طبيًا وسأساعدك بمعلومات دقيقة';
    document.getElementById('user-input').placeholder = language === 'en' ? 
        'Type your symptoms or question here...' : 
        'اكتب أعراضك أو استفسارك هنا...';
    
    const translations = {
        'ألم في الصدر': 'Chest pain',
        'ضيق تنفس': 'Shortness of breath',
        'دوخة': 'Dizziness',
        'خفقان القلب': 'Heart palpitations',
        'صداع': 'Headache',
        'غثيان': 'Nausea'
    };
    
    document.querySelectorAll('.quick-suggestion').forEach((button, index) => {
        const text = button.textContent;
        button.textContent = language === 'en' ? 
            translations[text] || text : 
            Object.keys(translations).find(key => translations[key] === text) || text;
    });
}

// Restart chat session
async function restartChat() {
    try {
        showLoadingAnimation();
        await fetch('/api/cleanup', { method: 'POST' });
        userId = null;
        sessionId = null;
        document.getElementById('chat-container').innerHTML = '';
        await startChat();
        toggleFAB();
    } catch (error) {
        hideLoadingAnimation();
        showModal(language === 'en' ? 'Error restarting chat' : 'خطأ أثناء إعادة بدء المحادثة');
        console.error('Error restarting chat:', error);
    }
}