"""
AI Social Media Content Generator - Streamlit Dashboard
Three screens: Input Form → Output Variants → Engagement Metrics
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Add repo root to path so modules/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from modules.predictor import predict_engagement
from modules.pipeline import generate_posts, optimize_variants
from modules.rag import build_index, retrieve_brand_context, is_index_built

# Initialize NLTK VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# Page configuration
st.set_page_config(
    page_title="AI Social Media Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .post-variant {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .score-high {
        background-color: #DEF7EC;
        color: #047857;
    }
    .score-medium {
        background-color: #FEF3C7;
        color: #B45309;
    }
    .score-low {
        background-color: #FEE2E2;
        color: #B91C1C;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screen' not in st.session_state:
    st.session_state.screen = 'input'
if 'generated_posts' not in st.session_state:
    st.session_state.generated_posts = None
if 'selected_post' not in st.session_state:
    st.session_state.selected_post = None

def run_pipeline(brand, topic, platform, tone, pdf_path=None):
    """
    Calls real Claude pipeline to generate 5 post variants.
    Returns list of variant dicts with post_text, hashtags,
    tone, suggested_posting_time, predicted_score, is_recommended.
    """

    # Generate 5 variants via Claude
    variants = generate_posts(
        brand_name=brand,
        topic=topic,
        tone=tone,
        platform=platform.lower(),
        pdf_path=pdf_path
    )

    if not variants:
        return []

    # Score and rank via XGBoost
    ranked, scores, scoring_failed = optimize_variants(
        variants,
        platform.lower()
    )

    # Add scoring_failed flag to each variant
    for v in ranked:
        v['scoring_failed'] = scoring_failed

    return ranked


def get_score_class(score, all_scores):
    """
    Returns CSS class based on relative score among all 5 variants.
    Top variant = green, bottom = red, rest = yellow.
    """
    if not all_scores or len(all_scores) < 2:
        return 'score-medium'
    max_score = max(all_scores)
    min_score = min(all_scores)
    if score == max_score:
        return 'score-high'
    elif score == min_score:
        return 'score-low'
    else:
        return 'score-medium'

# ============================================================================
# SCREEN 1: INPUT FORM
# ============================================================================

def screen_input():
    """Screen 1: Brand + Topic input form with platform selector and tone dropdown"""
    
    st.markdown('<p class="main-header">🤖 AI Social Media Content Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate platform-optimized content with AI-powered insights</p>', unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Content Configuration")
        
        # Brand input
        brand = st.text_input(
            "Brand Name",
            placeholder="e.g., EcoWear, TechCorp, FitFlow",
            help="Enter your brand or company name"
        )
        
        # Topic input
        topic = st.text_area(
            "Topic / Content Theme",
            placeholder="e.g., New sustainable product launch, Q3 results announcement, Industry insights on AI",
            help="What is this post about? Be specific.",
            height=100
        )
        
        # Platform selector
        platform = st.selectbox(
            "Target Platform",
            options=["Twitter", "LinkedIn", "Instagram"],
            help="Select the social media platform for content generation"
        )
        
        # Tone dropdown
        tone = st.selectbox(
            "Tone / Voice",
            options=["Professional", "Casual & Friendly", "Inspirational", "Educational", "Promotional"],
            help="Select the desired tone for your content"
        )

        st.markdown("---")
        st.markdown("### Brand Voice (Optional)")
        st.caption("Upload your brand guidelines PDF to align posts with your brand voice.")

        uploaded_pdf = st.file_uploader(
            "Upload Brand Guidelines PDF",
            type=["pdf"],
            help="PDF will be indexed using RAG to extract brand voice guidelines"
        )

        if uploaded_pdf:
            import tempfile
            if brand and not is_index_built(brand):
                with st.spinner(f"Building brand voice index for {brand}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_pdf.read())
                        tmp_path = tmp.name
                    st.session_state.pdf_path = tmp_path
                    build_index(tmp_path, brand)
                    st.success(f"✅ Brand voice indexed for {brand}!")
            elif brand and is_index_built(brand):
                st.success(f"✅ Brand voice already indexed for {brand}")
            else:
                st.warning("⚠️ Enter brand name above before uploading PDF")

    with col2:
        st.markdown("### Quick Tips")
        st.info("""
        **For Best Results:**
        - Be specific about your topic
        - Choose tone that matches your brand
        - Review all variants before selecting
        """)
        
        st.markdown("### Platform Guidelines")
        if platform == "Twitter":
            st.write("📝 **280 characters**")
            st.write("🏷️ 2-3 hashtags recommended")
        elif platform == "LinkedIn":
            st.write("📝 **1300-2000 characters**")
            st.write("🏷️ 3-5 hashtags recommended")
        else:  # Instagram
            st.write("📝 **2200 characters max**")
            st.write("🏷️ 5-8 hashtags recommended")
    
    st.markdown("---")
    
    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("🚀 Generate Content", type="primary", use_container_width=True):
            if not brand or not topic:
                st.error("⚠️ Please fill in both Brand Name and Topic")
            else:
                with st.spinner("Generating AI-powered content variants..."):
                    # Generate posts
                    posts = run_pipeline(brand, topic, platform, tone, pdf_path=st.session_state.get('pdf_path'))
                    
                    st.session_state.generated_posts = {
                        'brand': brand,
                        'topic': topic,
                        'platform': platform,
                        'tone': tone,
                        'variants': posts,
                        'generated_at': datetime.now()
                    }
                    st.session_state.screen = 'output'
                    st.rerun()
    
    with col_btn2:
        # Example button
        if st.button("📋 Load Example", use_container_width=True):
            st.session_state.example_loaded = True
            st.rerun()
    
    # Show example if button clicked
    if 'example_loaded' in st.session_state and st.session_state.example_loaded:
        st.markdown("---")
        st.markdown("### Example Loaded")
        st.success("""
        **Brand:** EcoWear  
        **Topic:** Launching new sustainable bamboo hoodie collection  
        **Platform:** Twitter  
        **Tone:** Casual & Friendly
        
        👆 Click 'Generate Content' to see AI-generated variants!
        """)

# ============================================================================
# SCREEN 2: OUTPUT VARIANTS
# ============================================================================

def screen_output():
    """Screen 2: Output panel showing 5 variants with predicted scores"""
    
    posts_data = st.session_state.generated_posts
    
    # Header with back button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">📝 Generated Content Variants</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sub-header">Platform: {posts_data["platform"]} | Tone: {posts_data["tone"]}</p>', unsafe_allow_html=True)
    with col2:
        if st.button("← Back to Input", use_container_width=True, key="output_back_btn"):
            st.session_state.screen = 'input'
            st.rerun()
        if st.button("📊 View Analytics", use_container_width=True, key="view_analytics_btn"):
            recommended = next(
                (v for v in posts_data['variants'] if v.get('is_recommended')),
                posts_data['variants'][0]
            )
            st.session_state.selected_post = recommended
            st.session_state.screen = 'metrics'
            st.rerun()
    
    # Show context
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Brand:** {posts_data['brand']}")
    with col_b:
        st.markdown(f"**Topic:** {posts_data['topic']}")
    
    st.markdown("---")
    st.markdown("### Select Your Preferred Variant")
    st.caption("AI has generated 5 variants ranked by predicted engagement score. Click 'View Analytics' to see detailed metrics.")
    
    # Display variants
    for idx, variant in enumerate(posts_data['variants'], 1):
        score = variant.get('predicted_score', 0) or 0
        all_scores = [v.get('predicted_score', 0) or 0 for v in posts_data['variants']]
        score_class = get_score_class(score, all_scores)
        
        with st.container():
            # Header row with variant number and score
            col_h1, col_h2 = st.columns([3, 1])
            
            with col_h1:
                st.markdown(f"#### Variant {idx}")
            with col_h2:
                recommended = "⭐ " if variant.get('is_recommended') else ""
                st.markdown(f'<span class="score-badge {score_class}">{recommended}Score: {score:,.0f}</span>', unsafe_allow_html=True)
            
            # Content
            st.markdown("**Content:**")
            st.text_area(
                "Content",
                value=variant.get('post_text', ''),
                height=150,
                key=f"content_{idx}",
                label_visibility="collapsed"
            )
            
            # Metadata
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.caption(f"📏 {len(variant.get('post_text', ''))} characters")
            with col_m2:
                st.caption(f"⏰ {variant.get('suggested_posting_time', 'N/A')}")
            with col_m3:
                hashtags = ' '.join(variant.get('hashtags', []))
                st.caption(f"🏷️ {hashtags if hashtags else 'No hashtags'}")
            
            # Action buttons
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                if st.button("📋 Copy", key=f"copy_{idx}", use_container_width=True):
                    st.success("✅ Copied to clipboard!")
            with col_b2:
                if st.button("✏️ Edit", key=f"edit_{idx}", use_container_width=True):
                    st.info("Edit mode enabled")
            with col_b3:
                if st.button("🔄 Regenerate", key=f"regen_{idx}", use_container_width=True):
                    st.info("Regenerating variant...")
            with col_b4:
                if st.button("✅ Select & Schedule", key=f"select_{idx}", type="primary", use_container_width=True):
                    st.success(f"✅ Variant {idx} selected for scheduling!")
            
            st.markdown("---")

# ============================================================================
# SCREEN 3: ENGAGEMENT METRICS
# ============================================================================

def screen_metrics():
    """Screen 3: Engagement metrics view with charts"""
    
    post_data = st.session_state.selected_post
    posts_context = st.session_state.generated_posts
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">📊 Engagement Analytics</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Predicted performance metrics and insights</p>', unsafe_allow_html=True)
    with col2:
        if st.button("← Back to Variants", use_container_width=True, key="metrics_back_btn"):
            st.session_state.screen = 'output'
            st.rerun()
        
    
    st.markdown("---")
    
    # Score and overview
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    score = post_data.get('predicted_score', 0) or 0
    features = post_data.get('features', {})
    scoring_failed = post_data.get('scoring_failed', False)

    platform = posts_context['platform'].lower()

    with col_s1:
        st.metric("Predicted Engagement", f"{score:,.0f}" if score else "N/A")
    with col_s2:
        st.metric("Sentiment Score",
                  f"{features.get('new_sentiment_score', 0):.2f}",
                  help="+1 very positive, -1 very negative")
    with col_s3:
        st.metric("Hashtag Count", features.get('hashtag_count', 0))
    with col_s4:
        st.metric("Character Count", features.get('caption_length', 0))

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### Feature Breakdown")
        feat_labels = ['Sentiment', 'Has CTA', 'Has Question', 'Has Emoji']
        feat_values = [
            max(0, features.get('new_sentiment_score', 0)),
            features.get('has_cta', 0),
            features.get('has_question', 0),
            features.get('has_emoji', 0)
        ]
        fig_pie = go.Figure(data=[go.Bar(
            x=feat_labels,
            y=feat_values,
            marker_color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
        )])
        fig_pie.update_layout(
            height=350,
            showlegend=False,
            margin=dict(t=20, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.markdown("### Variant Score Comparison")
        all_variants = posts_context['variants']
        var_labels = [f"Variant {i+1}" for i in range(len(all_variants))]
        var_scores = [v.get('predicted_score', 0) or 0 for v in all_variants]
        colors = ['#1E40AF' if v.get('is_recommended') else '#93C5FD'
                  for v in all_variants]
        fig_bar = go.Figure(go.Bar(
            x=var_labels,
            y=var_scores,
            marker_color=colors,
            text=[f"{s:,.0f}" for s in var_scores],
            textposition='outside'
        ))
        fig_bar.update_layout(
            yaxis_title="Predicted Engagement",
            height=350,
            showlegend=False,
            margin=dict(t=20, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Dark blue = recommended variant")

    st.markdown("---")
    st.markdown("### 🤖 AI Insights")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        has_cta = "✅ Has call-to-action" if features.get('has_cta') else "❌ No call-to-action"
        sentiment = features.get('new_sentiment_score', 0)
        sentiment_label = "✅ Positive tone" if sentiment > 0.2 else "⚠️ Neutral/negative tone"
        st.info(f"""
        **Post Analysis:**
        - {has_cta}
        - {sentiment_label} ({sentiment:.2f})
        - {features.get('hashtag_count', 0)} hashtags
        - {features.get('caption_length', 0)} characters
        """)

    with col_i2:
        from modules.predictor import get_best_posting_time
        best_time = get_best_posting_time(platform)
        st.warning(f"""
        **Suggestions:**
        - Best posting time: {best_time}
        - Add more emojis to boost Instagram/Twitter engagement
        - Test A/B variants to optimise performance
        """)

    st.markdown("---")
    col_a2, col_a3 = st.columns(2)

    with col_a2:
        if st.button("✏️ Edit Content", use_container_width=True, key="metrics_edit_btn"):
            st.session_state.screen = 'output'
            st.rerun()

    with col_a3:
        if st.button("🔄 Generate New", use_container_width=True, key="metrics_generate_btn"):
            st.session_state.screen = 'input'
            st.session_state.generated_posts = None
            st.rerun()
    
    with col_a2:
        if st.button("✏️ Edit Content", use_container_width=True):
            st.session_state.screen = 'output'
            st.rerun()
    
    with col_a3:
        if st.button("📥 Export Report", use_container_width=True):
            st.success("📥 Analytics report downloaded!")

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

def main():
    """Main app router"""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        current_screen = st.session_state.screen
        
        # Screen indicators
        screens = {
            'input': '1️⃣ Input Form',
            'output': '2️⃣ Content Variants',
            'metrics': '3️⃣ Analytics'
        }
        
        for key, label in screens.items():
            if key == current_screen:
                st.markdown(f"**→ {label}**")
            else:
                st.markdown(f"   {label}")
        
        st.markdown("---")
        
        # Quick stats (if data exists)
        if st.session_state.generated_posts:
            st.markdown("### 📊 Current Session")
            st.caption(f"**Platform:** {st.session_state.generated_posts['platform']}")
            st.caption(f"**Variants:** {len(st.session_state.generated_posts['variants'])}")
            st.caption(f"**Generated:** {st.session_state.generated_posts['generated_at'].strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 New Generation", use_container_width=True):
            st.session_state.screen = 'input'
            st.session_state.generated_posts = None
            st.session_state.selected_post = None
            st.rerun()
        
        st.markdown("---")
        st.caption("💡 **Tip:** All generated content uses AI and may need review before publishing.")
    
    # Route to appropriate screen
    if st.session_state.screen == 'input':
        screen_input()
    elif st.session_state.screen == 'output':
        if st.session_state.generated_posts:
            screen_output()
        else:
            st.session_state.screen = 'input'
            st.rerun()
    elif st.session_state.screen == 'metrics':
        if st.session_state.selected_post:
            screen_metrics()
        else:
            st.session_state.screen = 'output'
            st.rerun()

if __name__ == "__main__":
    main()
