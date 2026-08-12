import streamlit as st
import joblib
import time
import os

# 1. تهيئة إعدادات الصفحة والعنوان
st.set_page_config(
    page_title="Text Sentiment Analytics App",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Text Sentiment Analysis Production Engine")
st.markdown("""
This interactive web application deploys our **Production Champion ML Model** (TF-IDF + Logistic Regression Pipeline).
It analyzes raw English textual metrics live and predicts the underlying emotional state with low-latency execution.
""")
st.markdown("---")

# 2. دالة آمنة لتحميل الموديل وحفظه في الذاكرة التخزينية المؤقتة
@st.cache_resource
def load_champion_pipeline():
    # هنا قمنا بتعديل المسار ليتأكد من وجود الملف سواء كان في الفولدر أو في المجلد الرئيسي لكولاب
    possible_paths = ["production_assets/sentiment_pipeline_model.pkl", "sentiment_pipeline_model.pkl"]
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

pipeline = load_champion_pipeline()

# 3. واجهة المستخدم لإدخال النص
st.subheader("📝 Enter English Text for Analysis")
user_input = st.text_area(
    label="Type a sentence, review, or movie feedback below:",
    placeholder="Write your textual feedback here...",
    height=150
)

# 4. منطق التنبؤ والاستدلال عند الضغط على الزر
if st.button("🚀 Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please provide a valid text string input before running analysis.")
    elif pipeline is None:
        st.error("❌ Operational model pipeline asset (.pkl) is missing. Make sure it is saved in the directory.")
    else:
        with st.spinner("Processing text and extracting features..."):
            start_time = time.time()
            
            # تمرير النص للموديل (الـ Pipeline يتكفل بالـ TF-IDF والتصنيف معاً)
            prediction = pipeline.predict([user_input])[0]
            probabilities = pipeline.predict_proba([user_input])[0]
            
            latency_ms = (time.time() - start_time) * 1000
            
        st.success("✅ Inference Completed Successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.metric(label="Predicted Sentiment", value="🟢 Positive Emotions")
                st.balloons()
            else:
                st.metric(label="Predicted Sentiment", value="🔴 Negative Emotions")
                
        with col2:
            # حساب نسبة الثقة بناءً على الكلاس المتوقع
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            st.metric(label="Confidence Level", value=f"{confidence * 100:.2f}%")
            
        st.info(f"⚡ Deployment Latency Throughput: {latency_ms:.2f} ms")
