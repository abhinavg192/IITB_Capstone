"""
AI Social Media Content Generator - Streamlit Dashboard
Three screens: Input Form → Output Variants → Engagement Metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

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

# Mock AI generation function (replace with actual LLM call)
def generate_posts(brand, topic, platform, tone):
    """
    Mock function to generate social media posts
    In production, this would call GPT-4/Claude API
    """
    
    # Different variants with predicted scores
    variants = {
        'twitter': [
            {
                'content': f"Exciting news about {topic}! 🚀 {brand} is leading the way in innovation. What are your thoughts? #Innovation #{brand}",
                'score': 92,
                'hashtags': '#Innovation #' + brand,
                'char_count': 127,
                'best_time': '2:00 PM - 4:00 PM EST',
                'predicted_engagement': {'likes': 450, 'retweets': 120, 'replies': 35}
            },
            {
                'content': f"Big things happening at {brand}! Our latest work on {topic} is changing the game. Join the conversation 💪 #{brand}Community",
                'score': 88,
                'hashtags': '#' + brand + 'Community',
                'char_count': 134,
                'best_time': '11:00 AM - 1:00 PM EST',
                'predicted_engagement': {'likes': 380, 'retweets': 95, 'replies': 28}
            },
            {
                'content': f"Did you know? {brand}'s approach to {topic} is revolutionizing the industry. Learn more → [link] #TechNews #{brand}",
                'score': 85,
                'hashtags': '#TechNews #' + brand,
                'char_count': 125,
                'best_time': '9:00 AM - 11:00 AM EST',
                'predicted_engagement': {'likes': 320, 'retweets': 78, 'replies': 22}
            },
            {
                'content': f"🎯 {brand} + {topic} = Innovation at its finest. Here's what makes our approach different: [thread]",
                'score': 81,
                'hashtags': '',
                'char_count': 98,
                'best_time': '3:00 PM - 5:00 PM EST',
                'predicted_engagement': {'likes': 290, 'retweets': 65, 'replies': 18}
            },
            {
                'content': f"Transforming {topic} one step at a time. {brand} is committed to excellence. #Innovation #Excellence #{brand}",
                'score': 78,
                'hashtags': '#Innovation #Excellence #' + brand,
                'char_count': 115,
                'best_time': '5:00 PM - 7:00 PM EST',
                'predicted_engagement': {'likes': 250, 'retweets': 55, 'replies': 15}
            }
        ],
        'linkedin': [
            {
                'content': f"""I'm excited to share our latest insights on {topic}. 

At {brand}, we believe that innovation comes from understanding both the technical challenges and human needs. Our recent work demonstrates how a data-driven approach can transform outcomes.

Key takeaways:
• Strategic implementation matters more than technology alone
• Cross-functional collaboration drives success
• Continuous learning is essential

What's your experience with {topic}? I'd love to hear your thoughts.

#Innovation #Leadership #{brand}""",
                'score': 94,
                'hashtags': '#Innovation #Leadership #' + brand,
                'char_count': 456,
                'best_time': '8:00 AM - 10:00 AM EST',
                'predicted_engagement': {'likes': 680, 'comments': 45, 'shares': 120}
            },
            {
                'content': f"""The future of {topic} is here, and {brand} is at the forefront.

Our team has spent the last quarter developing solutions that address real-world challenges. The results speak for themselves:

→ Increased efficiency by 40%
→ Reduced time-to-market by 30%
→ Enhanced customer satisfaction scores

Proud of what we've built together. Here's to continued innovation!

#{brand} #Innovation #BusinessGrowth""",
                'score': 90,
                'hashtags': '#' + brand + ' #Innovation #BusinessGrowth',
                'char_count': 423,
                'best_time': '12:00 PM - 2:00 PM EST',
                'predicted_engagement': {'likes': 580, 'comments': 38, 'shares': 95}
            },
            {
                'content': f"""Sharing some reflections on {topic} and what it means for the industry.

At {brand}, we've learned that success requires:
1. Clear vision
2. Adaptable strategy  
3. Strong team collaboration

The landscape is evolving rapidly, and those who adapt will thrive.

What trends are you seeing in your field?

#ThoughtLeadership #{brand}""",
                'score': 86,
                'hashtags': '#ThoughtLeadership #' + brand,
                'char_count': 356,
                'best_time': '10:00 AM - 12:00 PM EST',
                'predicted_engagement': {'likes': 490, 'comments': 32, 'shares': 78}
            },
            {
                'content': f"""{brand} is transforming the way we approach {topic}.

Our methodology combines proven practices with innovative thinking. The results? Better outcomes, faster delivery, and more satisfied stakeholders.

Interested in learning more? Let's connect.

#Innovation #{brand}""",
                'score': 82,
                'hashtags': '#Innovation #' + brand,
                'char_count': 298,
                'best_time': '2:00 PM - 4:00 PM EST',
                'predicted_engagement': {'likes': 420, 'comments': 28, 'shares': 65}
            },
            {
                'content': f"""Quick update on our {topic} initiative at {brand}:

We're seeing tremendous progress and the team's dedication is inspiring. More details coming soon!

#{brand} #TeamWork #Progress""",
                'score': 75,
                'hashtags': '#' + brand + ' #TeamWork #Progress',
                'char_count': 198,
                'best_time': '4:00 PM - 6:00 PM EST',
                'predicted_engagement': {'likes': 340, 'comments': 22, 'shares': 48}
            }
        ],
        'instagram': [
            {
                'content': f"""✨ Big news from {brand}! ✨

We're thrilled to share our latest work on {topic}. Swipe to see behind-the-scenes moments from our team bringing this vision to life. 

From concept to reality, every step has been an incredible journey. Thank you to everyone who made this possible! 🙏

What would you like to see next? Drop your ideas in the comments! 👇

.
.
.
#Innovation #BehindTheScenes #{brand} #CreativeProcess #TeamWork #Vision #Progress #Community""",
                'score': 91,
                'hashtags': '#Innovation #BehindTheScenes #' + brand + ' #CreativeProcess #TeamWork #Vision #Progress #Community',
                'char_count': 512,
                'best_time': '11:00 AM - 1:00 PM EST',
                'predicted_engagement': {'likes': 1250, 'comments': 85, 'shares': 42}
            },
            {
                'content': f"""The future is here. 🚀

{brand} is reimagining {topic} and we couldn't be more excited about what's coming next.

Tag someone who needs to see this! 

#{brand} #Innovation #Future #Inspiration #Goals""",
                'score': 87,
                'hashtags': '#' + brand + ' #Innovation #Future #Inspiration #Goals',
                'char_count': 268,
                'best_time': '7:00 PM - 9:00 PM EST',
                'predicted_engagement': {'likes': 980, 'comments': 68, 'shares': 35}
            },
            {
                'content': f"""When innovation meets passion 💙

Our team at {brand} has been working hard on {topic}, and we can't wait to share more with you soon!

Stay tuned for updates. 

#{brand}Community #Innovation #ComingSoon #Excited""",
                'score': 83,
                'hashtags': '#' + brand + 'Community #Innovation #ComingSoon #Excited',
                'char_count': 289,
                'best_time': '5:00 PM - 7:00 PM EST',
                'predicted_engagement': {'likes': 820, 'comments': 54, 'shares': 28}
            },
            {
                'content': f"""Making waves in {topic} 🌊

{brand} is committed to excellence and innovation. Here's a glimpse of what we've been up to!

#{brand} #Excellence #Innovation""",
                'score': 79,
                'hashtags': '#' + brand + ' #Excellence #Innovation',
                'char_count': 198,
                'best_time': '3:00 PM - 5:00 PM EST',
                'predicted_engagement': {'likes': 680, 'comments': 42, 'shares': 22}
            },
            {
                'content': f"""New chapter, same dedication 📖

{brand} continues to push boundaries in {topic}. More updates coming!

#{brand} #Progress""",
                'score': 74,
                'hashtags': '#' + brand + ' #Progress',
                'char_count': 156,
                'best_time': '1:00 PM - 3:00 PM EST',
                'predicted_engagement': {'likes': 520, 'comments': 35, 'shares': 18}
            }
        ]
    }
    
    return variants.get(platform.lower(), variants['twitter'])

def get_score_class(score):
    """Return CSS class based on score"""
    if score >= 85:
        return 'score-high'
    elif score >= 75:
        return 'score-medium'
    else:
        return 'score-low'

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
                    # Simulate API call delay
                    import time
                    time.sleep(1.5)
                    
                    # Generate posts
                    posts = generate_posts(brand, topic, platform, tone)
                    
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
        if st.button("← Back to Input", use_container_width=True):
            st.session_state.screen = 'input'
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
        score = variant['score']
        score_class = get_score_class(score)
        
        with st.container():
            # Header row with variant number and score
            col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
            
            with col_h1:
                st.markdown(f"#### Variant {idx}")
            with col_h2:
                st.markdown(f'<span class="score-badge {score_class}">Score: {score}/100</span>', unsafe_allow_html=True)
            with col_h3:
                if st.button(f"📊 View Analytics", key=f"analytics_{idx}", use_container_width=True):
                    st.session_state.selected_post = variant
                    st.session_state.screen = 'metrics'
                    st.rerun()
            
            # Content
            st.markdown("**Content:**")
            st.text_area(
                "Content",
                value=variant['content'],
                height=150,
                key=f"content_{idx}",
                label_visibility="collapsed"
            )
            
            # Metadata
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.caption(f"📏 {variant['char_count']} characters")
            with col_m2:
                st.caption(f"⏰ Best time: {variant['best_time']}")
            with col_m3:
                st.caption(f"🏷️ {variant['hashtags'] if variant['hashtags'] else 'No hashtags'}")
            
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
        if st.button("← Back to Variants", use_container_width=True):
            st.session_state.screen = 'output'
            st.rerun()
    
    st.markdown("---")
    
    # Score and overview
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    score = post_data['score']
    engagement = post_data['predicted_engagement']
    
    platform = posts_context['platform'].lower()
    
    with col_s1:
        st.metric("Engagement Score", f"{score}/100", delta="8% above avg")
    
    if platform == 'twitter':
        with col_s2:
            st.metric("Predicted Likes", f"{engagement['likes']:,}", delta="+15%")
        with col_s3:
            st.metric("Predicted Retweets", f"{engagement['retweets']:,}", delta="+12%")
        with col_s4:
            st.metric("Predicted Replies", f"{engagement['replies']:,}", delta="+5%")
    elif platform == 'linkedin':
        with col_s2:
            st.metric("Predicted Likes", f"{engagement['likes']:,}", delta="+18%")
        with col_s3:
            st.metric("Predicted Comments", f"{engagement['comments']:,}", delta="+10%")
        with col_s4:
            st.metric("Predicted Shares", f"{engagement['shares']:,}", delta="+14%")
    else:  # instagram
        with col_s2:
            st.metric("Predicted Likes", f"{engagement['likes']:,}", delta="+20%")
        with col_s3:
            st.metric("Predicted Comments", f"{engagement['comments']:,}", delta="+16%")
        with col_s4:
            st.metric("Predicted Shares", f"{engagement['shares']:,}", delta="+8%")
    
    st.markdown("---")
    
    # Two columns for charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### Predicted Engagement Breakdown")
        
        # Engagement pie chart
        if platform == 'twitter':
            labels = ['Likes', 'Retweets', 'Replies']
            values = [engagement['likes'], engagement['retweets'], engagement['replies']]
        elif platform == 'linkedin':
            labels = ['Likes', 'Comments', 'Shares']
            values = [engagement['likes'], engagement['comments'], engagement['shares']]
        else:
            labels = ['Likes', 'Comments', 'Shares']
            values = [engagement['likes'], engagement['comments'], engagement['shares']]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=['#3B82F6', '#10B981', '#F59E0B'])
        )])
        
        fig_pie.update_layout(
            showlegend=True,
            height=350,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_chart2:
        st.markdown("### Engagement Over Time (Predicted)")
        
        # Time series prediction
        hours = list(range(24))
        base_engagement = sum(values)
        
        # Simulate engagement curve
        engagement_curve = []
        for h in hours:
            if 9 <= h <= 17:  # Peak hours
                engagement_curve.append(base_engagement * 0.8 + random.randint(-50, 100))
            elif 18 <= h <= 22:  # Evening
                engagement_curve.append(base_engagement * 0.5 + random.randint(-30, 80))
            else:  # Off hours
                engagement_curve.append(base_engagement * 0.2 + random.randint(-20, 50))
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=hours,
            y=engagement_curve,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#3B82F6', width=2),
            name='Predicted Engagement'
        ))
        
        fig_time.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Engagement Count",
            height=350,
            margin=dict(t=20, b=0, l=0, r=0),
            showlegend=False
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Comparison with other variants
    st.markdown("---")
    st.markdown("### Variant Comparison")
    
    # Create comparison data
    all_variants = posts_context['variants']
    comparison_data = {
        'Variant': [f'Variant {i+1}' for i in range(len(all_variants))],
        'Score': [v['score'] for v in all_variants],
        'Engagement': [sum(v['predicted_engagement'].values()) for v in all_variants]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Bar chart comparison
    fig_compare = go.Figure()
    
    fig_compare.add_trace(go.Bar(
        x=df_comparison['Variant'],
        y=df_comparison['Score'],
        name='Engagement Score',
        marker_color='#3B82F6'
    ))
    
    fig_compare.update_layout(
        xaxis_title="Variant",
        yaxis_title="Score",
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # AI Insights
    st.markdown("---")
    st.markdown("### 🤖 AI-Generated Insights")
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.info(f"""
        **Strengths:**
        - High engagement score ({score}/100)
        - Optimal character length for {posts_context['platform']}
        - Strong call-to-action present
        - Appropriate hashtag usage
        """)
    
    with col_i2:
        st.warning(f"""
        **Suggestions:**
        - Consider posting at {post_data['best_time']} for maximum reach
        - Add more visual elements (emoji/images) to increase engagement
        - Test A/B variants to optimize performance
        """)
    
    # Action buttons
    st.markdown("---")
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        if st.button("✅ Approve & Schedule", type="primary", use_container_width=True):
            st.success(f"✅ Post scheduled for {post_data['best_time']}!")
    
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
