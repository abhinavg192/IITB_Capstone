"""
AI Social Media Content Generator - Streamlit Dashboard
Three screens: Input Form → Output Variants → Engagement Metrics
"""
import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import tempfile

# Must be set before all other imports
# Prevents libomp conflicts on Mac ARM
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Add repo root to path so modules/ is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Real pipeline imports
try:
    from modules.pipeline import generate_posts, optimize_variants
    from modules.rag import build_index, retrieve_brand_context, is_index_built
    from modules.predictor import get_best_posting_time, get_feature_importance
    PIPELINE_AVAILABLE = True
except Exception as e:
    PIPELINE_AVAILABLE = False
    PIPELINE_ERROR = str(e)

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
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_uploaded_pdf(uploaded_file) -> str:
    """
    Saves Streamlit uploaded PDF to a temp file.
    Returns the temp file path for RAG indexing.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def format_score(score) -> str:
    """Format predicted engagement score for display."""
    if score is None:
        return "N/A"
    return f"{score:,.0f}"


def get_top_features(variant: dict) -> list:
    """
    Returns top 3 feature explanations for a variant.
    Used in the 'why this post scored well' display.
    """
    features = variant.get('features', {})
    explanations = []

    if features.get('has_cta'):
        explanations.append("✅ Has a clear call-to-action")
    if features.get('new_sentiment_score', 0) > 0.3:
        explanations.append("✅ Positive sentiment tone")
    if features.get('has_question'):
        explanations.append("✅ Contains a question (drives comments)")
    if features.get('has_emoji'):
        explanations.append("✅ Uses emojis (boosts engagement)")

    platform = features.get('platform_encoded', 0)
    hashtags = features.get('hashtag_count', 0)
    if platform == 0 and 2 <= hashtags <= 3:
        explanations.append("✅ Optimal hashtag count for Twitter")
    elif platform == 1 and 3 <= hashtags <= 5:
        explanations.append("✅ Optimal hashtag count for LinkedIn")
    elif platform == 2 and 5 <= hashtags <= 10:
        explanations.append("✅ Optimal hashtag count for Instagram")

    return explanations[:3] if explanations else ["Post features scored well overall"]


# ============================================================================
# SCREEN 1: INPUT FORM
# ============================================================================

def screen_input():
    """Screen 1: Brand + Topic input form with PDF upload"""

    st.markdown('<p class="main-header">🤖 AI Social Media Content Generator</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate platform-optimized content with AI-powered insights</p>',
                unsafe_allow_html=True)

    # Pipeline unavailable warning
    if not PIPELINE_AVAILABLE:
        st.error(f"⚠️ Pipeline not available: {PIPELINE_ERROR}")
        st.info("Check that all modules are installed and API keys are set in .env")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Content Configuration")

        brand = st.text_input(
            "Brand Name",
            placeholder="e.g., Adobe, Adidas, Duolingo",
            help="Enter your brand or company name"
        )

        topic = st.text_area(
            "Topic / Content Theme",
            placeholder="e.g., New sustainable product launch, Q3 results, Industry insights on AI",
            help="What is this post about? Be specific.",
            height=100
        )

        platform = st.selectbox(
            "Target Platform",
            options=["LinkedIn", "Twitter", "Instagram"],
            help="Select the social media platform"
        )

        tone = st.selectbox(
            "Tone / Voice",
            options=[
                "Professional",
                "Casual & Friendly",
                "Inspirational",
                "Educational",
                "Promotional",
                "Bold & Direct"
            ],
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
            if brand and not is_index_built(brand):
                with st.spinner(f"Building brand voice index for {brand}..."):
                    pdf_path = save_uploaded_pdf(uploaded_pdf)
                    st.session_state.pdf_path = pdf_path
                    build_index(pdf_path, brand)
                    st.success(f"✅ Brand voice indexed for {brand}!")
            elif brand and is_index_built(brand):
                st.success(f"✅ Brand voice already indexed for {brand}")
            else:
                st.warning("Enter brand name above to index guidelines")

    with col2:
        st.markdown("### Quick Tips")
        st.info("""
        **For Best Results:**
        - Be specific about your topic
        - Upload brand guidelines PDF
        - Choose tone that matches your brand
        - Review all variants before selecting
        """)

        st.markdown("### Platform Guidelines")
        if platform == "Twitter":
            st.write("📝 **280 characters max**")
            st.write("🏷️ 2-3 hashtags recommended")
            st.write("⏰ Best: Wed 9:00 AM")
        elif platform == "LinkedIn":
            st.write("📝 **150-300 words**")
            st.write("🏷️ 3-5 hashtags recommended")
            st.write("⏰ Best: Tue 9:00 AM")
        else:
            st.write("📝 **2200 characters max**")
            st.write("🏷️ 5-10 hashtags recommended")
            st.write("⏰ Best: Wed 11:00 AM")

        st.markdown("### How It Works")
        st.markdown("""
        1. 📄 RAG retrieves brand context
        2. 🤖 Claude generates 5 variants
        3. 📊 XGBoost scores each variant
        4. ⭐ Top variant recommended
        """)

    st.markdown("---")

    col_btn1, col_btn2, _ = st.columns([1, 1, 2])

    with col_btn1:
        generate_clicked = st.button(
            "🚀 Generate Content",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        if st.button("📋 Load Example", use_container_width=True):
            st.session_state.example_loaded = True
            st.rerun()

    if 'example_loaded' in st.session_state and st.session_state.example_loaded:
        st.info("""
        **Example loaded:**
        Brand: Adobe | Topic: Adobe's acquisition of Semrush |
        Platform: LinkedIn | Tone: Professional

        Fill in the fields above and click Generate!
        """)

    if generate_clicked:
        if not brand or not topic:
            st.error("⚠️ Please fill in both Brand Name and Topic")
        else:
            with st.spinner("🤖 Generating content variants..."):
                try:
                    variants = generate_posts(
                        brand_name=brand,
                        topic=topic,
                        tone=tone,
                        platform=platform.lower(),
                        pdf_path=st.session_state.pdf_path
                    )
                    if not variants:
                        st.error("❌ Generation failed — check your API key and try again")
                        return
                except Exception as e:
                    st.error(f"❌ Generation error: {e}")
                    return

            with st.spinner("📊 Scoring variants with XGBoost..."):
                try:
                    ranked, scores, scoring_failed = optimize_variants(
                        variants,
                        platform.lower()
                    )
                except Exception as e:
                    ranked = variants
                    scores = []
                    scoring_failed = True

            st.session_state.generated_posts = {
                'brand':          brand,
                'topic':          topic,
                'platform':       platform,
                'tone':           tone,
                'variants':       ranked,
                'scoring_failed': scoring_failed,
                'generated_at':   datetime.now()
            }
            st.session_state.screen = 'output'
            st.rerun()


# ============================================================================
# SCREEN 2: OUTPUT VARIANTS
# ============================================================================

def screen_output():
    """Screen 2: 5 ranked post variants with scores"""

    posts_data = st.session_state.generated_posts
    scoring_failed = posts_data.get('scoring_failed', False)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">📝 Generated Content Variants</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<p class="sub-header">Platform: {posts_data["platform"]} '
            f'| Brand: {posts_data["brand"]}</p>',
            unsafe_allow_html=True
        )
    with col2:
        if st.button("← Back to Input", use_container_width=True):
            st.session_state.screen = 'input'
            st.rerun()

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Topic:** {posts_data['topic']}")
    with col_b:
        st.markdown(f"**Tone:** {posts_data['tone']}")

    st.markdown("---")

    # Scoring status banner
    if scoring_failed:
        st.warning(
            "⚠️ Engagement scoring unavailable — posts shown in generation order. "
            "You can still read and select the post you prefer."
        )
    else:
        st.success("✅ Posts ranked by predicted engagement score. Top post recommended.")

    st.markdown("### Select Your Preferred Variant")

    for idx, variant in enumerate(posts_data['variants']):
        is_recommended = variant.get('is_recommended', False)
        score = variant.get('predicted_score')

        # Highlight recommended variant
        if is_recommended:
            st.markdown("---")
            st.markdown(
                '<span class="recommended-badge">⭐ RECOMMENDED — Highest Predicted Engagement</span>',
                unsafe_allow_html=True
            )

        with st.container():
            col_h1, col_h2, col_h3 = st.columns([2, 1, 1])

            with col_h1:
                rank_label = f"Rank {idx+1}" if not scoring_failed else f"Variant {idx+1}"
                st.markdown(f"#### {rank_label}")

            with col_h2:
                if not scoring_failed and score is not None:
                    st.markdown(
                        f'<span class="score-label">Score: {format_score(score)}</span>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("Score: N/A")

            with col_h3:
                if st.button(
                    "📊 View Analytics",
                    key=f"analytics_{idx}",
                    use_container_width=True
                ):
                    st.session_state.selected_post = variant
                    st.session_state.screen = 'metrics'
                    st.rerun()

            # Post content
            st.markdown("**Post:**")
            st.text_area(
                "Post content",
                value=variant.get('post_text', ''),
                height=150,
                key=f"content_{idx}",
                label_visibility="collapsed"
            )

            # Metadata
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                char_count = len(variant.get('post_text', ''))
                st.caption(f"📏 {char_count} characters")
            with col_m2:
                posting_time = variant.get('suggested_posting_time', 'N/A')
                st.caption(f"⏰ {posting_time}")
            with col_m3:
                hashtags = ' '.join(variant.get('hashtags', []))
                st.caption(f"🏷️ {hashtags if hashtags else 'No hashtags'}")

            # Reasoning (COT output)
            if variant.get('reasoning'):
                with st.expander("💭 View AI reasoning"):
                    st.caption(variant['reasoning'])

            # Action buttons
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                if st.button("📋 Copy", key=f"copy_{idx}", use_container_width=True):
                    st.success("✅ Copied!")
            with col_b2:
                if st.button("🔄 Regenerate", key=f"regen_{idx}", use_container_width=True):
                    st.info("Click Generate on input screen to regenerate")
            with col_b3:
                if st.button(
                    "✅ Select This Post",
                    key=f"select_{idx}",
                    type="primary" if is_recommended else "secondary",
                    use_container_width=True
                ):
                    st.success(f"✅ Post selected! Posting time: {posting_time}")

        if not is_recommended:
            st.markdown("---")

# ============================================================================
# SCREEN 3: ENGAGEMENT METRICS
# ============================================================================

def screen_metrics():
    """Screen 3: Engagement analytics for selected variant"""

    post_data     = st.session_state.selected_post
    posts_context = st.session_state.generated_posts
    platform      = posts_context['platform'].lower()
    scoring_failed = posts_context.get('scoring_failed', False)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">📊 Engagement Analytics</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Predicted performance metrics and insights</p>',
                    unsafe_allow_html=True)
    with col2:
        if st.button("← Back to Variants", use_container_width=True):
            st.session_state.screen = 'output'
            st.rerun()

    st.markdown("---")

    # ── Metrics row ──
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    score    = post_data.get('predicted_score')
    features = post_data.get('features', {})

    with col_s1:
        if score is not None:
            st.metric("Predicted Engagement", format_score(score))
        else:
            st.metric("Predicted Engagement", "N/A")

    with col_s2:
        st.metric(
            "Sentiment Score",
            f"{features.get('new_sentiment_score', 0):.2f}",
            help="VADER sentiment: +1 very positive, -1 very negative"
        )

    with col_s3:
        st.metric(
            "Hashtag Count",
            features.get('hashtag_count', 0)
        )

    with col_s4:
        best_time = get_best_posting_time(platform)
        st.metric("Best Posting Time", best_time)

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### Variant Score Comparison")

        all_variants = posts_context['variants']

        if not scoring_failed:
            labels = [f"Variant {i+1}" for i in range(len(all_variants))]
            scores = [v.get('predicted_score', 0) or 0 for v in all_variants]
            colors = ['#1E40AF' if v.get('is_recommended') else '#93C5FD'
                      for v in all_variants]

            fig = go.Figure(go.Bar(
                x=labels,
                y=scores,
                marker_color=colors,
                text=[format_score(s) for s in scores],
                textposition='outside'
            ))
            fig.update_layout(
                yaxis_title="Predicted Engagement Score",
                height=350,
                showlegend=False,
                margin=dict(t=20, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Dark blue = recommended variant")
        else:
            st.info("Scoring unavailable — chart not available")

    with col_chart2:
        st.markdown("### Feature Importance")

        try:
            importance_df = get_feature_importance()
            top5 = importance_df.head(5)

            fig2 = go.Figure(go.Bar(
                x=top5['Importance'],
                y=top5['Feature'],
                orientation='h',
                marker_color='#3B82F6'
            ))
            fig2.update_layout(
                xaxis_title="Importance",
                height=350,
                showlegend=False,
                margin=dict(t=20, b=0, l=0, r=0)
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.info("Feature importance chart unavailable")

    # ── Why this post scored well ──
    st.markdown("---")
    st.markdown("### 🤖 Why This Post Scored Well")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        explanations = get_top_features(post_data)
        for exp in explanations:
            st.markdown(exp)

    with col_i2:
        st.markdown("**Optimal Posting Time:**")
        st.info(f"📅 {get_best_posting_time(platform)}")

        st.markdown("**Post Preview:**")
        st.text_area(
            "Selected post",
            value=post_data.get('post_text', ''),
            height=150,
            label_visibility="collapsed",
            disabled=True
        )

    # ── Action buttons ──
    st.markdown("---")
    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        if st.button("✅ Approve & Use", type="primary", use_container_width=True):
            st.success(f"✅ Post approved! Best time: {get_best_posting_time(platform)}")

    with col_a2:
        if st.button("← Choose Different", use_container_width=True):
            st.session_state.screen = 'output'
            st.rerun()

    with col_a3:
        if st.button("🔄 Generate New", use_container_width=True):
            st.session_state.screen = 'input'
            st.session_state.generated_posts = None
            st.rerun()

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
            data = st.session_state.generated_posts
            st.markdown("### 📊 Current Session")
            st.caption(f"**Brand:** {data['brand']}")
            st.caption(f"**Platform:** {data['platform']}")
            st.caption(f"**Variants:** {len(data['variants'])}")
            st.caption(f"**Generated:** {data['generated_at'].strftime('%H:%M:%S')}")

            if not data.get('scoring_failed'):
                top = next((v for v in data['variants'] if v.get('is_recommended')), None)
                if top and top.get('predicted_score'):
                    st.caption(f"**Top Score:** {format_score(top['predicted_score'])}")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 New Generation", use_container_width=True):
            st.session_state.screen = 'input'
            st.session_state.generated_posts = None
            st.session_state.selected_post = None
            st.rerun()
        
        st.markdown("---")
        st.caption("Built with Claude + LangChain + XGBoost")
        st.caption("IITB Capstone 2026")
    
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
