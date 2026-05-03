# AI Social Media Content Generator - Streamlit Dashboard

A professional 3-screen dashboard for AI-powered social media content generation.

## 📋 Features

### Screen 1: Input Form
- Brand name input
- Topic/theme text area
- Platform selector (Twitter, LinkedIn, Instagram)
- Tone dropdown (Professional, Casual, Inspirational, etc.)
- Generate and Example buttons
- Platform-specific guidelines

### Screen 2: Output Variants
- 5 AI-generated content variants
- Engagement score for each variant (0-100)
- Character count and best posting time
- Copy, Edit, Regenerate, and Select buttons
- View Analytics button for each variant

### Screen 3: Engagement Metrics
- Predicted engagement metrics (likes, retweets/comments/shares)
- Engagement breakdown pie chart
- Time-based engagement prediction
- Variant comparison bar chart
- AI-generated insights and suggestions
- Approve, Edit, and Export actions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📱 Navigation

The dashboard has 3 screens with smooth navigation:

1. **Input Form** → Enter brand, topic, platform, and tone
2. **Content Variants** → Review 5 AI-generated variants with scores
3. **Engagement Analytics** → View detailed metrics and insights

## 🎨 UI Features

- **Professional Design**: Clean, modern interface with custom CSS
- **Responsive Layout**: Works on different screen sizes
- **Interactive Charts**: Plotly-powered visualizations
- **Color-Coded Scores**: Green (85+), Yellow (75-84), Red (<75)
- **Sidebar Navigation**: Easy screen tracking and quick stats

## 🔧 Customization

### To Connect Real AI (GPT-4/Claude):

Replace the `generate_posts()` function in `app.py` with actual API calls:

```python
def generate_posts(brand, topic, platform, tone):
    # Replace with actual LLM API call
    from anthropic import Anthropic
    
    client = Anthropic(api_key="your-key")
    
    prompt = f"""Generate a {platform} post for {brand} about {topic} 
    in a {tone} tone..."""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Process and return variants
    return variants
```

### To Add More Platforms:

Add to the platform selector and create appropriate templates in `generate_posts()`.

## 📊 Mock Data

Currently uses mock data to demonstrate functionality:
- Generates 5 variants per request
- Assigns engagement scores (74-94 range)
- Predicts metrics based on platform
- Simulates best posting times

## 🎯 For Your Project

This dashboard is perfect for:
- **Project demos** - Show working prototype
- **User testing** - Get feedback on UI/UX
- **Development** - Test with real API later
- **Presentation** - Professional interface for proposals

## 📝 Notes

- All generated content is currently **mock data**
- Replace `generate_posts()` with real LLM API for production
- Engagement predictions are simulated for demonstration
- Charts and metrics update based on generated content

## 🐛 Troubleshooting

**Issue:** Dashboard doesn't start  
**Solution:** Make sure Streamlit is installed: `pip install streamlit`

**Issue:** Charts don't display  
**Solution:** Install plotly: `pip install plotly`

**Issue:** Navigation broken  
**Solution:** Clear browser cache or use incognito mode

## 📦 Project Structure

```
.
├── app.py              # Main Streamlit dashboard
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Next Steps

1. Run the dashboard to test the UI
2. Integrate real LLM API (GPT-4 or Claude)
3. Add brand guidelines loading (RAG)
4. Connect to actual social media APIs (optional)
5. Add user authentication (if needed)

---

**Built with:** Streamlit, Plotly, Pandas  
**Purpose:** AI/ML Project - Social Media Content Generation  
**Status:** Proof of Concept Dashboard
