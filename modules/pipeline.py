"""
modules/pipeline.py
AI Social Media Content Generation Pipeline
Extracted from notebooks/abhinav/prompt_template_pipeline.ipynb

Provides:
    - generate_posts()     → generates 5 post variants via Claude
    - optimize_variants()  → scores and ranks variants via XGBoost
    - get_examples()       → returns few-shot examples per platform

Author: Abhinav Gupta | IITB Capstone 2026
"""

import os
import re
import json
import sys

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# Must be set before any other imports to prevent
# libomp conflicts on Mac ARM (torch + faiss + xgboost)
# ─────────────────────────────────────────────────────────────

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ─────────────────────────────────────────────────────────────
# API KEY LOADING
# ─────────────────────────────────────────────────────────────

def _load_api_keys():
    """Load API keys from .env file in repo root."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
    env_path = os.path.join(repo_root, '.env')
    load_dotenv(dotenv_path=env_path)

    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. "
            "Check your .env file in the repo root."
        )
    return anthropic_key

# ─────────────────────────────────────────────────────────────
# LLM INITIALIZATION
# ─────────────────────────────────────────────────────────────

def _init_llm(model: str = "claude-haiku-4-5-20251001") -> ChatAnthropic:
    """
    Initialize Claude LLM.
    Default: claude-haiku-4-5-20251001 (dev/cost efficient)
    Production: claude-sonnet-4-6 (higher quality)
    """
    api_key = _load_api_keys()
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        max_tokens=4000
    )

# Initialize once at module load
_llm = _init_llm()

# ─────────────────────────────────────────────────────────────
# RAG MODULE IMPORT
# ─────────────────────────────────────────────────────────────

try:
    from modules.rag import (
        build_index,
        retrieve_brand_context,
        is_index_built
    )
    RAG_AVAILABLE = True
except ImportError:
    try:
        # Fallback for when running from repo root
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from modules.rag import (
            build_index,
            retrieve_brand_context,
            is_index_built
        )
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PREDICTOR MODULE IMPORT
# ─────────────────────────────────────────────────────────────

try:
    from modules.predictor import predict_engagement
except ImportError:
    def predict_engagement(features_dict):
        raise RuntimeError(
            "predictor module not found. "
            "Ensure modules/predictor.py exists."
        )

# ─────────────────────────────────────────────────────────────
# VADER SENTIMENT (for feature extraction)
# ─────────────────────────────────────────────────────────────

import ssl
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Fix SSL certificate issue on Mac
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download VADER if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

_vader = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES LIBRARY
# Sources: Nike (Twitter), Adobe (LinkedIn), NatGeo (Instagram)
# Real posts from real brands — not fabricated
# ─────────────────────────────────────────────────────────────

_TWITTER_EXAMPLES = """
EXAMPLE 1 (Athlete/Event Celebration - Nike):
"When you're this fast, you don't ask for permission. Jutta Leerdam
breaks the Olympic record in the Speed Skating 1000m and wins her
first Gold. #MilanoCortina2026 #Olympics"

EXAMPLE 2 (Product Launch - Nike):
"Mute the gallery. Introducing Nike Powerbeats Pro 2, the ultimate
fitness earbuds. The first-ever Beats x Nike collab — where premium
sound meets unbeatable stability. No advice required.
Launching March 20th at 9am PST."

EXAMPLE 3 (Motivational/Quote - Nike):
"There's no failure in sports. It's steps to success." - @Giannis_An34
Regardless of the outcome, there's always a reward ahead. #AlwaysForward
"""

_LINKEDIN_EXAMPLES = """
EXAMPLE 1 (Thought Leadership - Adobe):
"Creative teams are under pressure to deliver more content across more
channels than ever before. Many are turning to AI to handle production
tasks like resizing, versioning, and early-stage ideation, so they can
spend more time refining concepts and craft. Adobe's latest research
breaks down how this shift is changing the way we do creative work."

EXAMPLE 2 (Corporate Announcement - Adobe):
"The way brands are discovered is changing fast. Today, Adobe completed
its acquisition of Semrush, expanding how businesses show up, get found,
and drive growth in an AI-first world. AI-driven traffic to U.S. retail
sites is up 269% year over year, yet most businesses still have significant
gaps in how they appear across AI surfaces."

EXAMPLE 3 (Community/Human Story - Adobe):
"My greatest inspiration came from being raised by a self-made single
mom who showed us strength and resilience. From drawing at a young age
to now utilizing Adobe tools to bridge the gap between art and business,
Alexandra Yvette continues to create and share her talents with the
world this Women's History Month."
"""

_INSTAGRAM_EXAMPLES = """
EXAMPLE 1 (Wildlife/Action Story - NatGeo):
"This image was taken near Polán, Toledo province (Spain) last August,
from a tiny hide-out overlooking a small waterhole where lynxes
occasionally come down to drink. It was an extremely hot afternoon,
and a rabbit was very close to the water. Suddenly, a lynx appeared
silently, but the rabbit noticed it at the very last second.
Photo by @alexandrovich_yo from our @natgeoyourshot community."

EXAMPLE 2 (Landscape/Philosophical - NatGeo):
"From Punta Helbronner, 11,370 feet above sea level on the Italian
side of Mont Blanc in the Alps, my guide walks out across the snow
toward Dent du Géant, the Giant's Tooth, illuminated only by the
light of my drone during a long exposure. Photos by @reuben"

EXAMPLE 3 (Milestone/Achievement - NatGeo):
"Barcelona's Sagrada Família has reached a new milestone. With the
cross now installed atop its central Jesus tower, the basilica stands
more than 560 feet (170m) — surpassing Germany's Ulm Minster as the
world's tallest church. Photographs by Nuria Puentes, National Geographic-España"
"""

def get_examples(platform: str) -> str:
    """
    Returns few-shot examples for the given platform.

    Args:
        platform: "twitter", "linkedin", or "instagram"

    Returns:
        str: formatted examples string for prompt injection
    """
    mapping = {
        "twitter":   _TWITTER_EXAMPLES,
        "linkedin":  _LINKEDIN_EXAMPLES,
        "instagram": _INSTAGRAM_EXAMPLES
    }
    return mapping.get(platform.lower(), "")


# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# Three levels to demonstrate optimization progression
# ─────────────────────────────────────────────────────────────

# Level 1 — Zero-shot (baseline, no examples, no COT)
ZEROSHOT_TEMPLATE = PromptTemplate(
    input_variables=["brand_name", "topic", "tone", "platform"],
    template="""
You are an expert social media content creator.

Generate exactly 5 distinct social media post variants for the brand {brand_name}.

PLATFORM: {platform}
TOPIC: {topic}
TONE: {tone}

PLATFORM RULES:
- Twitter: maximum 280 characters, 2-3 hashtags, punchy and direct
- LinkedIn: 150-300 words, professional but human, 3-5 hashtags
- Instagram: storytelling style, 5-10 hashtags, strong opening hook

OUTPUT FORMAT:
Return exactly 5 posts numbered like this:
1. [post text] [hashtags]
2. [post text] [hashtags]
3. [post text] [hashtags]
4. [post text] [hashtags]
5. [post text] [hashtags]

Write only the posts. No explanations.
"""
)

# Level 2 — Few-shot (with real brand examples)
FEWSHOT_TEMPLATE = PromptTemplate(
    input_variables=["brand_name", "topic", "tone", "platform", "examples"],
    template="""
You are an expert social media content creator.

Generate exactly 5 distinct social media post variants for the brand {brand_name}.

PLATFORM: {platform}
TOPIC: {topic}
TONE: {tone}

PLATFORM RULES:
- Twitter: maximum 280 characters, 2-3 hashtags, punchy and direct
- LinkedIn: 150-300 words, professional but human, 3-5 hashtags
- Instagram: storytelling style, 5-10 hashtags, strong opening hook

HIGH PERFORMING EXAMPLES FOR REFERENCE:
{examples}

Now write 5 posts matching the same quality and style as the examples above.

OUTPUT FORMAT:
Return exactly 5 posts numbered like this:
1. [post text] [hashtags]
2. [post text] [hashtags]
3. [post text] [hashtags]
4. [post text] [hashtags]
5. [post text] [hashtags]

Write only the posts. No explanations.
"""
)

# Level 3 — Unified (few-shot + COT + JSON) — PRODUCTION TEMPLATE
UNIFIED_TEMPLATE = PromptTemplate(
    input_variables=["brand_name", "topic", "tone", "platform",
                     "examples", "brand_context"],
    template="""
You are an expert social media content creator specializing in brand voice alignment.

Generate exactly 5 distinct social media post variants for the brand {brand_name}.

PLATFORM: {platform}
TOPIC: {topic}
TONE: {tone}

BRAND GUIDELINES FOR {brand_name}:
{brand_context}

PLATFORM RULES:
- Twitter: maximum 280 characters, 2-3 hashtags, punchy and direct
- LinkedIn: 150-300 words, professional but human, 3-5 hashtags, short paragraphs
- Instagram: storytelling style, strong opening hook, 5-10 hashtags, emojis throughout

HIGH PERFORMING EXAMPLES FOR REFERENCE:
{examples}

THINK STEP BY STEP BEFORE WRITING EACH VARIANT:
Step 1 - Audience: Who is the primary audience for this variant?
Step 2 - Key Message: What is the single most important message?
Step 3 - Tone Check: Does the tone match the brand and platform?
Step 4 - Hook: What opening line will stop the scroll?
Step 5 - CTA: What action should the reader take?

Make each variant structurally distinct:
- Variant 1: Lead with business impact
- Variant 2: Lead with customer benefit
- Variant 3: Lead with a bold statement or question
- Variant 4: Lead with data or proof points
- Variant 5: Lead with a human or emotional angle

IMPORTANT: Respond with ONLY a valid JSON array.
No explanations. No preamble. No markdown. Just the JSON.

Return exactly this structure:
[
  {{
    "variant_id": 1,
    "reasoning": "brief explanation of thinking for this variant",
    "post_text": "the post content here",
    "hashtags": ["#hashtag1", "#hashtag2"],
    "tone": "describe the tone used",
    "suggested_posting_time": "best day and time to post",
    "platform": "{platform}"
  }},
  {{
    "variant_id": 2,
    "reasoning": "brief explanation of thinking for this variant",
    "post_text": "the post content here",
    "hashtags": ["#hashtag1", "#hashtag2"],
    "tone": "describe the tone used",
    "suggested_posting_time": "best day and time to post",
    "platform": "{platform}"
  }},
  {{
    "variant_id": 3,
    "reasoning": "brief explanation of thinking for this variant",
    "post_text": "the post content here",
    "hashtags": ["#hashtag1", "#hashtag2"],
    "tone": "describe the tone used",
    "suggested_posting_time": "best day and time to post",
    "platform": "{platform}"
  }},
  {{
    "variant_id": 4,
    "reasoning": "brief explanation of thinking for this variant",
    "post_text": "the post content here",
    "hashtags": ["#hashtag1", "#hashtag2"],
    "tone": "describe the tone used",
    "suggested_posting_time": "best day and time to post",
    "platform": "{platform}"
  }},
  {{
    "variant_id": 5,
    "reasoning": "brief explanation of thinking for this variant",
    "post_text": "the post content here",
    "hashtags": ["#hashtag1", "#hashtag2"],
    "tone": "describe the tone used",
    "suggested_posting_time": "best day and time to post",
    "platform": "{platform}"
  }}
]
"""
)


# ─────────────────────────────────────────────────────────────
# JSON PARSER
# ─────────────────────────────────────────────────────────────

def _parse_json_response(response_text: str) -> list:
    """
    Cleans and parses Claude's JSON response.
    Handles markdown code blocks if present.
    """
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def _get_vader_score(text: str) -> float:
    """
    VADER sentiment score — used for XGBoost feature.
    Named new_sentiment_score to match Manas's model schema.
    Returns compound score from -1 to 1.
    NOTE: Uses VADER not DistilBERT because Manas trained
    XGBoost on VADER scores — must match training distribution.
    """
    return round(_vader.polarity_scores(text)['compound'], 4)


def extract_features(variant: dict, platform: str) -> dict:
    """
    Extracts content features matching Manas's XGBoost schema exactly.

    Args:
        variant: single post dict from generate_posts()
        platform: "twitter", "linkedin", or "instagram"

    Returns:
        dict with keys matching Manas's model training features
    """
    post_text = variant['post_text']
    hashtags  = variant['hashtags']
    posting_time = variant.get('suggested_posting_time', '9:00 AM')

    # Caption length
    caption_length = len(post_text)

    # Hashtag count
    hashtag_count = len(hashtags)

    # VADER sentiment — named new_sentiment_score to match Manas
    new_sentiment_score = _get_vader_score(post_text)

    # Has CTA — matches Manas's training CTA keywords
    cta_keywords = [
        'link', 'register', 'learn more', 'sign up', 'click here',
        'visit', 'download', 'bio', 'get started', 'click',
        'discover', 'explore', 'try', 'get', 'join', 'shop',
        'buy', 'check out', 'read', 'watch', 'follow', 'share'
    ]
    has_cta = int(any(kw in post_text.lower() for kw in cta_keywords))

    # Platform encoded — named platform_encoded to match Manas
    platform_map = {"twitter": 0, "linkedin": 1, "instagram": 2}
    platform_encoded = platform_map.get(platform.lower(), 0)

    # Hour posted — required by Manas's model
    # Extracted from suggested_posting_time field
    hour_match = re.search(r'(\d+):\d+\s*(AM|PM)', posting_time)
    if hour_match:
        hour = int(hour_match.group(1))
        if hour_match.group(2) == 'PM' and hour != 12:
            hour += 12
        elif hour_match.group(2) == 'AM' and hour == 12:
            hour = 0
    else:
        hour = 9  # default 9am

    # Additional features for display / future model use
    word_count   = len(post_text.split())
    has_question = int('?' in post_text)
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F9FF"
        u"\U00002700-\U000027BF" "]+",
        flags=re.UNICODE
    )
    has_emoji = int(bool(emoji_pattern.search(post_text)))

    return {
        # Core features — match Manas's model exactly
        "caption_length":      caption_length,
        "hashtag_count":       hashtag_count,
        "new_sentiment_score": new_sentiment_score,
        "has_cta":             has_cta,
        "platform_encoded":    platform_encoded,
        "hour_posted":         hour,
        # Additional features
        "word_count":          word_count,
        "has_question":        has_question,
        "has_emoji":           has_emoji
    }


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def generate_posts(
    brand_name: str,
    topic: str,
    tone: str,
    platform: str,
    pdf_path: str = None,
    brand_context: str = None,
    model: str = None
) -> list:
    """
    Generates 5 social media post variants using the unified
    template (few-shot + chain-of-thought + JSON + RAG).

    Args:
        brand_name (str):    e.g. "Adobe"
        topic (str):         e.g. "Adobe's acquisition of Semrush"
        tone (str):          e.g. "professional, forward looking"
        platform (str):      "twitter", "linkedin", or "instagram"
        pdf_path (str):      optional path to brand voice PDF
        brand_context (str): optional pre-retrieved brand context string
        model (str):         optional Claude model override

    Returns:
        list: 5 structured post variants, each containing:
            - variant_id
            - reasoning
            - post_text
            - hashtags
            - tone
            - suggested_posting_time
            - platform
    """
    global _llm

    # Swap model if requested
    if model:
        _llm = _init_llm(model)

    examples = get_examples(platform)

    # ── Step 1: Get brand context ──
    if brand_context:
        context = brand_context

    elif pdf_path and RAG_AVAILABLE:
        if not is_index_built(brand_name):
            build_index(pdf_path, brand_name)
        else:
            print(f"✅ Using existing RAG index for {brand_name}")

        context = retrieve_brand_context(
            brand_name=brand_name,
            topic=topic,
            platform=platform,
            tone=tone
        )
        print(f"✅ Retrieved brand context ({len(context)} chars)")

    else:
        context = (
            f"Write in a {tone} tone appropriate "
            f"for {brand_name} on {platform}."
        )

    # ── Step 2: Assemble prompt ──
    filled_prompt = UNIFIED_TEMPLATE.format(
        brand_name=brand_name,
        topic=topic,
        tone=tone,
        platform=platform,
        examples=examples,
        brand_context=context
    )

    # ── Step 3: Call Claude ──
    response = _llm.invoke([HumanMessage(content=filled_prompt)])

    # ── Step 4: Parse and return ──
    return _parse_json_response(response.content)


def optimize_variants(variants: list, platform: str) -> tuple:
    """
    Scores and ranks 5 post variants by predicted engagement.

    Uses Manas's XGBoost model via predict_engagement().
    All variants scored at neutral hour=9am so posting time
    does not influence content quality ranking.

    If scoring fails — returns posts unranked with scoring_failed=True.
    Users can still read and select the post they prefer.

    Args:
        variants: list of 5 post dicts from generate_posts()
        platform: "twitter", "linkedin", or "instagram"

    Returns:
        ranked_variants: list sorted best first (or original order)
        scores:          list of raw scores (or empty list)
        scoring_failed:  bool — True if scoring unavailable
    """
    scores = []

    try:
        print("Extracting features and scoring variants...")

        for i, variant in enumerate(variants):
            features = extract_features(variant, platform)

            # Fix hour to neutral — posting time is a recommendation,
            # not a content quality signal
            features['hour_posted'] = 9

            score = predict_engagement(features)
            scores.append(score)
            variant['predicted_score'] = score
            variant['is_recommended']  = False
            variant['scoring_failed']  = False
            print(f"  Variant {i+1}: score = {score}")

        # Sort descending
        ranked = sorted(
            variants,
            key=lambda x: x['predicted_score'],
            reverse=True
        )
        ranked[0]['is_recommended'] = True

        print(
            f"\n⭐ Recommended: Variant {ranked[0]['variant_id']} "
            f"(score: {ranked[0]['predicted_score']})"
        )
        return ranked, scores, False

    except Exception as e:
        print(f"\n⚠️ Scoring unavailable: {e}")
        print("   Returning all posts unscored.")
        print("   You can still read and select the post you prefer.")

        for variant in variants:
            variant['predicted_score'] = None
            variant['is_recommended']  = False
            variant['scoring_failed']  = True

        return variants, [], True
