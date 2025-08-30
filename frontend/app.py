# frontend/app.py

import streamlit as st
from streamlit import components
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import io
import base64
import requests
import time
import os
import sys
import numpy as np

try:
    from PIL.UnidentifiedImageError import UnidentifiedImageError
except ImportError:
    # Fallback for older Pillow versions
    UnidentifiedImageError = Exception
import openai

# Import drawing utilities
from drawing_utils import image_to_bytes, bytes_to_image


# --- Analysis Animation Functions ---
def create_analysis_animation():
    """Create CSS for the blue wave analysis animation"""
    return """
    <style>
    @keyframes analysis-wave {
        0% {
            transform: translateY(0px) scaleY(1);
            opacity: 0.3;
        }
        25% {
            transform: translateY(-10px) scaleY(1.2);
            opacity: 0.6;
        }
        50% {
            transform: translateY(0px) scaleY(1);
            opacity: 0.8;
        }
        75% {
            transform: translateY(10px) scaleY(0.8);
            opacity: 0.6;
        }
        100% {
            transform: translateY(0px) scaleY(1);
            opacity: 0.3;
        }
    }
    
    .analysis-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, 
            rgba(0, 123, 255, 0.1) 0%, 
            rgba(0, 123, 255, 0.3) 25%, 
            rgba(0, 123, 255, 0.1) 50%, 
            rgba(0, 123, 255, 0.3) 75%, 
            rgba(0, 123, 255, 0.1) 100%);
        border-radius: 10px;
        pointer-events: none;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        backdrop-filter: blur(2px);
        width: 100%;
        height: 100%;
    }
    
    .analysis-wave {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(0, 123, 255, 0.4) 20%, 
            rgba(0, 123, 255, 0.8) 50%, 
            rgba(0, 123, 255, 0.4) 80%, 
            transparent 100%);
        animation: analysis-wave 2s ease-in-out infinite;
        transform-origin: center;
    }
    
    .analysis-text {
        position: relative;
        z-index: 1001;
        color: #0066cc;
        font-weight: bold;
        font-size: 18px;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
        background: rgba(255, 255, 255, 0.9);
        padding: 10px 20px;
        border-radius: 25px;
        border: 2px solid #0066cc;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
    }
    
    .analysis-container {
        position: relative;
        display: block;
        width: 100%;
        height: 100%;
    }
    
    .canvas-container {
        position: relative;
        display: block;
        width: 100%;
    }
    
    .stImage {
        position: relative;
    }
    
    .stImage > img {
        position: relative;
        z-index: 1;
    }
    </style>
    """


def show_analysis_animation(container, text="AI Analyzing Sketch..."):
    """Display the blue wave analysis animation over the sketch"""
    container.markdown(create_analysis_animation(), unsafe_allow_html=True)

    # Create the analysis overlay
    analysis_html = f"""
    <div class="analysis-container">
        <div class="analysis-overlay">
            <div class="analysis-wave"></div>
            <div class="analysis-text">{text}</div>
        </div>
    </div>
    """

    container.markdown(analysis_html, unsafe_allow_html=True)


# --- AI Analysis Functions (Moved to top for global access) ---


def analyze_sketch_with_qwen(image_bytes, mode="auto", sidebar_selections=None):
    """
    Analyze sketch using Qwen2.5-VL-32B to generate intelligent prompts and parameters.

    Args:
        image_bytes: The sketch image (drawn or uploaded)
        mode: "auto" for full automation, "guided" for user-guided analysis
        sidebar_selections: Dict containing user's sidebar selections (art_style, sketch_type, theme, subject_type, etc.)

    Returns:
        dict: Analysis results with prompts, parameters, and recommendations
    """
    try:
        # Convert image to base64 for API
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Extract theme and subject type from sidebar selections
        theme = (
            sidebar_selections.get("theme", "portrait")
            if sidebar_selections
            else "portrait"
        )
        subject_type = (
            sidebar_selections.get("subject_type", "auto")
            if sidebar_selections
            else "auto"
        )
        art_style = (
            sidebar_selections.get("art_style", "realistic")
            if sidebar_selections
            else "realistic"
        )
        sketch_type = (
            sidebar_selections.get("sketch_type", "full_body")
            if sidebar_selections
            else "full_body"
        )

        if mode == "auto":
            # Full automatic analysis
            system_prompt = f"""You are an expert AI sketch analyzer for a drawing-to-painting application. Your job is to analyze sketches and generate detailed prompts for AI image generation.

CRITICAL: ALWAYS ANALYZE WHAT IS ACTUALLY DRAWN IN THE SKETCH FIRST, regardless of user's sidebar selections. The sketch content takes priority over sidebar defaults.

ANALYSIS PRIORITY ORDER:
1. FIRST: Analyze what is actually drawn in the sketch (mountains, person, object, etc.)
2. SECOND: Apply appropriate art style and parameters based on the actual content
3. THIRD: Consider user's sidebar preferences as secondary guidance

DETAILED ANALYSIS GUIDELINES:

LANDSCAPE/MOUNTAIN ANALYSIS (HIGH PRIORITY):
- If you see ANY triangular shapes, pointed peaks, or mountain-like forms → This is a LANDSCAPE
- Count the EXACT number of mountains/peaks visible
- Note their relative heights and positions (foreground, middle ground, background)
- Identify mountain shapes (sharp peaks, rounded summits, jagged ridges)
- Observe spacing between peaks (clustered, scattered, single range)
- Note any distinctive features (snow caps, rock formations, vegetation)
- Count visible valleys, passes, or gaps between mountains
- Identify the overall mountain range structure (linear, clustered, scattered)
- If mountains detected → Generate LANDSCAPE prompt, ignore portrait defaults

HUMAN FIGURE ANALYSIS (HIGHEST PRIORITY FOR PORTRAITS):
- **GENDER DETECTION (CRITICAL):**
  * FEMALE indicators: long hair (any length past shoulders), lipstick, eyelashes, dresses, skirts, jewelry, feminine facial features, rounded jawline, softer features
  * MALE indicators: short hair, mustache, beard, masculine jawline, suits, ties, broader shoulders, angular features
  * If unclear, analyze hair length, facial structure, and clothing style for context clues

- **HAIR DETAILS (HIGH PRIORITY):**
  * Hair length: short, medium, long, very long
  * Hair style: straight, wavy, curly, braided, ponytail, bun, loose, styled
  * Hair color: blonde, brunette, black, red, auburn, gray, white, dyed colors
  * Hair texture: smooth, voluminous, thin, thick, layered, bangs
  * Hair accessories: clips, headbands, hats, ribbons, flowers

- **FACIAL DETAILS (HIGH PRIORITY):**
  * Expression: smile, serious, neutral, happy, sad, surprised, confident, shy
  * Eye details: eye color, eyelashes, eye shape, makeup, glasses
  * Facial features: nose shape, lip shape, cheekbones, jawline
  * Skin tone: fair, medium, dark, olive, warm, cool undertones
  * Age indicators: youthful, mature, elderly, specific age range

- **CLOTHING & STYLE:**
  * Clothing type: casual, formal, business, elegant, sporty, vintage
  * Specific items: dresses, suits, jeans, shirts, jackets, accessories
  * Style: modern, classic, trendy, bohemian, minimalist, luxurious
  * Colors and patterns: solid colors, prints, stripes, floral, geometric

- **POSE & BODY LANGUAGE:**
  * Pose: standing, sitting, walking, portrait, full body, three-quarter view
  * Body language: confident, relaxed, formal, casual, dynamic, static
  * Hand positions: hands in pockets, crossed arms, pointing, gesturing
  * Head position: straight, tilted, looking up/down, profile, frontal

OBJECT/SCENE DETECTION:
- If figure doesn't look human (stickman, geometric shapes):
  * Triangles/pointed shapes → mountain, pyramid, roof
  * Connected rectangles → house, building, architecture
  * Multiple geometric shapes → cityscape, landmark, structure
  * Organic curves → landscape, nature scene
- **TREAT STICKMAN DRAWINGS AS REAL HUMANS** with proper anatomy and clothing

Current user selections (use as secondary guidance only):
- Art Style: {art_style}
- Sketch Type: {sketch_type}
- Subject Type: {subject_type}
- Theme: {theme}

ANALYSIS PROCESS:
1. FIRST: Identify what is actually drawn (mountain, person, object, landscape, etc.)
2. SECOND: If mountains/landscape detected → Generate LANDSCAPE prompt with mountain details
3. THIRD: If person detected → Generate PORTRAIT prompt with human details
4. FOURTH: Apply appropriate art style and parameters for the detected content
5. FIFTH: Consider user's sidebar preferences as style guidance

IMPORTANT: 
- If you see mountains → Generate LANDSCAPE prompt regardless of sidebar theme
- If you see a person → Generate PORTRAIT prompt regardless of sidebar theme
- The sketch content ALWAYS determines the subject type, not the sidebar defaults
- Use sidebar selections only for art style guidance, not content determination
- **For human figures, prioritize hair details, facial expressions, and gender indicators**
- **Even simple stick figures should be treated as real humans with detailed descriptions**

Generate a detailed, natural paragraph prompt that accurately describes what you see in the sketch."""

            user_prompt = f"""The user has selected {art_style} style, {sketch_type} type, {subject_type} subject, and {theme} theme. 

Analyze this sketch and provide ONLY a detailed, natural paragraph prompt that matches their {art_style} {sketch_type} selection for {theme} theme.

IMPORTANT: 
- If it's a stickman, treat it as a REAL HUMAN BEING with proper anatomy and describe the exact pose you see
- If it's a landscape or object, describe it accurately with specific details
- Write the prompt as a natural, flowing paragraph (NOT JSON) that could be used directly for AI image generation
- Make it detailed, descriptive, and professional
- Focus on visual elements, composition, lighting, and atmosphere
- Use the mountain analysis guidelines to count peaks, describe positioning, shapes, and features

Write ONLY the prompt paragraph - no JSON, no additional fields, just the descriptive text."""

        else:  # guided mode
            # Use sidebar selections automatically
            art_style = sidebar_selections.get("art_style", "realistic")
            sketch_type = sidebar_selections.get("sketch_type", "full_body")

            # Customize subject type based on user selection
            subject_description = ""
            if subject_type == "girl":
                subject_description = "young female"
            elif subject_type == "boy":
                subject_description = "young male"
            elif subject_type == "woman":
                subject_description = "adult female"
            elif subject_type == "man":
                subject_description = "adult male"
            elif subject_type == "person":
                subject_description = "gender-neutral person"
            else:
                subject_description = "person"

            system_prompt = f"""You are an expert art analyst specializing in {theme} themes. The user has selected:
- Art Style: {art_style}
- Sketch Type: {sketch_type}
- Subject Type: {subject_type}
- Theme: {theme}

Analyze this sketch and provide:
1. What the subject is (person, object, scene)
2. If it's a person: specific details about pose, clothing, hair, etc. matching {subject_description}
3. A strong, detailed prompt that matches their selected {art_style} style and {sketch_type} type for {theme} theme
4. Optimal parameters for their chosen mode
5. Suggestions for improvement

IMPORTANT: If this is a stickman or simple line drawing, treat it as a REAL HUMAN BEING. Describe the pose, proportions, and body language as if it were a detailed human figure. The stickman represents a real person's pose and should be converted to a realistic human with proper anatomy, clothing, and features while preserving the exact pose and positioning. Ensure the generated prompt respects the user's theme and chosen subject type {subject_type}.

Focus on making the prompt match the sketch while working within their chosen parameters and {theme} theme."""

            user_prompt = f"""The user has selected {art_style} style, {sketch_type} type, {subject_type} subject, and {theme} theme. 

Analyze this sketch and provide ONLY a detailed, natural paragraph prompt that matches their {art_style} {sketch_type} selection for {theme} theme.

IMPORTANT: 
- If it's a stickman, treat it as a REAL HUMAN BEING with proper anatomy and describe the exact pose you see
- If it's a landscape or object, describe it accurately with specific details
- Write the prompt as a natural, flowing paragraph (NOT JSON) that could be used directly for AI image generation
- Make it detailed, descriptive, and professional
- Focus on visual elements, composition, lighting, and atmosphere
- Use the mountain analysis guidelines to count peaks, describe positioning, shapes, and features

Write ONLY the prompt paragraph - no JSON, no additional fields, just the descriptive text."""

        # Call Qwen2.5-VL via OpenRouter
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
            default_headers={
                "HTTP-Referer": "https://drawing-to-painting-app.com",
                "X-Title": "Drawing to Painting App",
            },
        )

        response = client.chat.completions.create(
            model="qwen/qwen2.5-vl-32b-instruct:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        # Parse the response
        analysis_text = response.choices[0].message.content

        # Since we're now requesting paragraph responses, create structured data directly
        analysis_data = {
            "subject": (
                "landscape"
                if any(
                    word in analysis_text.lower()
                    for word in ["mountain", "peak", "landscape", "nature", "forest"]
                )
                else "person"
            ),
            "person_details": {"gender": "unknown", "pose": "standing"},
            "recommended_style": art_style,
            "recommended_sketch_type": sketch_type,
            "generated_prompt": analysis_text.strip(),
            "negative_prompt": "deformed, blurry, low quality, unrealistic, cartoon, anime",
            "parameter_suggestions": f"Use {art_style} style with {sketch_type} focus for {theme} theme",
        }

        return analysis_data

    except Exception as e:
        st.error(f"❌ Sketch analysis failed: {str(e)}")
        st.info("💡 Falling back to manual input mode")
        return None


def auto_generate_from_sketch(image_bytes, sidebar_selections=None):
    """
    Fully automatic mode: AI handles everything using sidebar selections if available.
    """
    st.info("🤖 Auto Mode: AI is configuring everything automatically...")

    # Analyze the sketch
    analysis = analyze_sketch_with_qwen(
        image_bytes, mode="auto", sidebar_selections=sidebar_selections
    )

    if not analysis:
        st.error("❌ AI analysis failed. Please use manual mode.")
        return None

    # Display analysis results
    st.success("✅ AI Analysis Complete!")

    with st.expander("🔍 AI Analysis Results", expanded=True):
        # Show analysis results in robotic terminal style
        st.markdown(
            """
        <style>
        .analysis-results-expander {
            background: #000000;
            border: 2px solid #00ff00;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            color: #00ff00;
            line-height: 1.6;
        }
        .analysis-item {
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #00ff00;
        }
        .analysis-label {
            color: #00ff00;
            font-weight: bold;
            margin-right: 10px;
        }
        .analysis-data {
            color: #ffffff;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Display key analysis results
        if "subject" in analysis:
            st.markdown(
                f'<div class="analysis-item"><span class="analysis-label">🤖 SUBJECT:</span><span class="analysis-data">{analysis["subject"]}</span></div>',
                unsafe_allow_html=True,
            )

        if "generated_prompt" in analysis:
            st.markdown(
                f'<div class="analysis-item"><span class="analysis-label">🎨 PROMPT:</span><span class="analysis-data">{analysis["generated_prompt"]}</span></div>',
                unsafe_allow_html=True,
            )

        if "recommendations" in analysis:
            st.markdown(
                f'<div class="analysis-item"><span class="analysis-label">💡 TIPS:</span><span class="analysis-data">{analysis["recommendations"]}</span></div>',
                unsafe_allow_html=True,
            )

        # Show full JSON in a collapsible section for developers
        with st.expander("🔧 Raw Analysis Data (for developers)", expanded=False):
            st.json(analysis)

    # Auto-select parameters based on AI recommendations
    if "recommended_style" in analysis:
        st.session_state.art_style = analysis["recommended_style"]
    if "recommended_sketch_type" in analysis:
        st.session_state.sketch_type = analysis["recommended_sketch_type"]

    # Auto-generate prompt
    if "generated_prompt" in analysis:
        st.session_state.auto_prompt = analysis["generated_prompt"]

    # Auto-select optimal parameters
    if "optimal_parameters" in analysis:
        st.session_state.auto_params = analysis["optimal_parameters"]

    st.success("🎯 AI has automatically configured all parameters!")
    st.info("💡 Click 'Generate Painting' to create your masterpiece!")

    return analysis


def guided_generate_from_sketch(image_bytes, sidebar_selections):
    """
    Guided mode: Uses user's sidebar selections, AI provides object suggestions and generates prompts.
    """
    st.info("🎯 Guided Mode: Using your selections with AI-generated prompts...")

    # Analyze the sketch with user's sidebar selections
    analysis = analyze_sketch_with_qwen(
        image_bytes, mode="guided", sidebar_selections=sidebar_selections
    )

    if not analysis:
        st.error("❌ AI analysis failed. Please use manual mode.")
        return None

    # Display analysis results
    st.success("✅ AI Analysis Complete!")

    with st.expander("🔍 AI Analysis Results", expanded=True):
        # Show analysis results in robotic terminal style
        st.markdown(
            """
        <style>
        .analysis-results-expander-guided {
            background: #000000;
            border: 2px solid #38ef7d;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            color: #38ef7d;
            line-height: 1.6;
        }
        .analysis-item-guided {
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #38ef7d;
        }
        .analysis-label-guided {
            color: #38ef7d;
            font-weight: bold;
            margin-right: 10px;
        }
        .analysis-data-guided {
            color: #ffffff;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Display key analysis results
        if "subject" in analysis:
            st.markdown(
                f'<div class="analysis-item-guided"><span class="analysis-label-guided">🎯 SUBJECT:</span><span class="analysis-data-guided">{analysis["subject"]}</span></div>',
                unsafe_allow_html=True,
            )

        if "generated_prompt" in analysis:
            st.markdown(
                f'<div class="analysis-item-guided"><span class="analysis-label-guided">🎨 PROMPT:</span><span class="analysis-data-guided">{analysis["generated_prompt"]}</span></div>',
                unsafe_allow_html=True,
            )

        if "recommendations" in analysis:
            st.markdown(
                f'<div class="analysis-item-guided"><span class="analysis-label-guided">💡 TIPS:</span><span class="analysis-data-guided">{analysis["recommendations"]}</span></div>',
                unsafe_allow_html=True,
            )

        # Show full JSON in a collapsible section for developers
        with st.expander("🔧 Raw Analysis Data (for developers)", expanded=False):
            st.json(analysis)

    # Generate guided prompt
    if "generated_prompt" in analysis:
        st.session_state.guided_prompt = analysis["generated_prompt"]

    # Provide parameter suggestions
    if "parameter_suggestions" in analysis:
        st.info(f"💡 AI Suggestions: {analysis['parameter_suggestions']}")

    st.success("🎯 AI has generated the perfect prompt for your selections!")
    st.info("💡 Click 'Generate Painting' to create your masterpiece!")

    return analysis


def test_qwen_integration():
    """
    Test function to verify Qwen integration setup without making API calls.
    """
    try:
        import openai

        st.success("✅ OpenAI package imported successfully")

        # Test configuration
        if OPENROUTER_API_KEY and OPENROUTER_BASE_URL:
            st.success("✅ OpenRouter configuration loaded")
            st.info(f"🔑 API Key: {OPENROUTER_API_KEY[:20]}...")
            st.info(f"🌐 Base URL: {OPENROUTER_BASE_URL}")
        else:
            st.error("❌ OpenRouter configuration missing")

        return True
    except ImportError as e:
        st.error(f"❌ Import error: {e}")
        st.info("💡 Install required packages: pip install openai")
        return False
    except Exception as e:
        st.error(f"❌ Configuration error: {e}")
        return False


# --- End of AI Analysis Functions ---

# Add parent directory to path to import modules from root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ngrok_config import get_backend_url, get_base_url, print_current_config

# get_backend_url() will return the single backend URL for all modes
# print_current_config() will print the current ngrok URL


# Add a reusable curtain reveal with gold frame and enhanced sparkles
def show_curtain_reveal(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    html = f"""
    <style>
      body {{ margin:0; background: radial-gradient(circle at 30% 20%, #2b2e4a 0%, #1b1e34 45%, #0f1228 100%); }}
      .stage {{ position:relative; max-width:820px; margin:0 auto; padding:24px; }}
      .curtain {{ position:absolute; top:0; width:50%; height:100%; background: linear-gradient(90deg, #5a2e2e, #8b3b3b); box-shadow: inset 0 0 40px rgba(0,0,0,.5); z-index:10; transition: transform 4.5s cubic-bezier(0.25, 0.46, 0.45, 0.94); }}
      .left {{ left:0; transform: translateX(0%); }}
      .right {{ right:0; transform: translateX(0%); }}
      .frame {{ position:relative; border-radius:18px; padding:14px; background: linear-gradient(145deg, #cda349, #a77b22); box-shadow: 0 6px 22px rgba(0,0,0,.35), inset 0 0 2px rgba(255,255,255,.5); }}
      .inner {{ border-radius:12px; overflow:hidden; background:#000; box-shadow: inset 0 0 12px rgba(0,0,0,.4); }}
      .inner img {{ width:100%; height:460px; object-fit:cover; display:block; }}
      
      /* Enhanced Sparkling Effects */
      .sparkle {{ position:absolute; width:8px; height:8px; background: radial-gradient(circle, #fff, rgba(255,255,255,0)); border-radius:50%; filter: drop-shadow(0 0 12px #ffd86b); animation: twinkle 3s infinite ease-in-out; }}
      .sparkle:nth-child(1) {{ top:18px; left:18px; animation-delay: .2s; }}
      .sparkle:nth-child(2) {{ top:12px; right:22px; animation-delay: .5s; }}
      .sparkle:nth-child(3) {{ bottom:20px; left:30px; animation-delay: 1s; }}
      .sparkle:nth-child(4) {{ bottom:26px; right:28px; animation-delay: 1.3s; }}
      .sparkle:nth-child(5) {{ top:50%; left:15px; animation-delay: .8s; }}
      .sparkle:nth-child(6) {{ top:50%; right:15px; animation-delay: 1.1s; }}
      .sparkle:nth-child(7) {{ top:25%; left:50%; animation-delay: .4s; }}
      .sparkle:nth-child(8) {{ bottom:25%; left:50%; animation-delay: .7s; }}
      
      /* Floating sparkles around the frame */
      .floating-sparkle {{ position:absolute; width:4px; height:4px; background: radial-gradient(circle, #ffd86b, rgba(255,216,107,0)); border-radius:50%; filter: drop-shadow(0 0 8px #ffd86b); animation: float 4s infinite ease-in-out; }}
      .floating-sparkle:nth-child(9) {{ top:-10px; left:20%; animation-delay: 0s; }}
      .floating-sparkle:nth-child(10) {{ top:-15px; right:30%; animation-delay: 1s; }}
      .floating-sparkle:nth-child(11) {{ bottom:-10px; left:40%; animation-delay: 2s; }}
      .floating-sparkle:nth-child(12) {{ bottom:-15px; right:20%; animation-delay: 3s; }}
      
      @keyframes twinkle {{ 
        0%,100% {{ opacity:.3; transform: scale(.7) rotate(0deg); filter: drop-shadow(0 0 8px #ffd86b); }}
        25% {{ opacity:.8; transform: scale(1.1) rotate(90deg); filter: drop-shadow(0 0 15px #ffd86b); }}
        50% {{ opacity:1; transform: scale(1.3) rotate(180deg); filter: drop-shadow(0 0 20px #ffd86b); }}
        75% {{ opacity:.6; transform: scale(1.1) rotate(270deg); filter: drop-shadow(0 0 12px #ffd86b); }}
      }}
      
      @keyframes float {{
        0%,100% {{ transform: translateY(0px) scale(1); opacity: 0.7; }}
        25% {{ transform: translateY(-8px) scale(1.2); opacity: 1; }}
        50% {{ transform: translateY(-15px) scale(1.1); opacity: 0.8; }}
        75% {{ transform: translateY(-8px) scale(1.3); opacity: 0.9; }}
      }}
      
      /* Golden glow effect */
      .frame::before {{
        content: '';
        position: absolute;
        top: -2px; left: -2px; right: -2px; bottom: -2px;
        background: linear-gradient(45deg, #ffd86b, #cda349, #ffd86b, #a77b22);
        border-radius: 20px;
        z-index: -1;
        animation: glow 3s ease-in-out infinite alternate;
      }}
      
      @keyframes glow {{
        0% {{ opacity: 0.5; filter: blur(3px); }}
        100% {{ opacity: 0.8; filter: blur(5px); }}
      }}
    </style>
    <div class="stage"> 
      <div class="curtain left"></div>
      <div class="curtain right"></div>
      <div class="frame">
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="floating-sparkle"></div>
        <div class="floating-sparkle"></div>
        <div class="floating-sparkle"></div>
        <div class="floating-sparkle"></div>
        <div class="inner">
          <img src="data:image/png;base64,{img_b64}"/>
        </div>
      </div>
    </div>
    <script>
      // Dramatic curtain reveal with longer delay
      setTimeout(()=>{{
        const L=document.querySelector('.left'); 
        const R=document.querySelector('.right');
        if(L&&R){{
          // Add a dramatic pause before opening
          L.style.transform='translateX(-100%)'; 
          R.style.transform='translateX(100%)';
        }}
      }}, 2500); // Increased delay to 2.5 seconds for maximum suspense
    </script>
    """
    components.html(html, height=540)


# Import the enhanced theme system
try:
    from enhanced_theme_system import (
        get_theme_info,
        detect_theme_style,
        get_enhanced_prompts_for_theme,
        THEME_CATEGORIES,
    )

    ENHANCED_THEME_SYSTEM_AVAILABLE = True
except ImportError:
    ENHANCED_THEME_SYSTEM_AVAILABLE = False
    print("⚠️ Enhanced theme system not available, using fallback functions")

    # Fallback functions
    def get_theme_info(theme_name):
        # Theme-specific descriptions for fallback
        theme_descriptions = {
            # Realistic themes
            "Realistic portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful": {
                "description": "Professional photography with natural skin texture and detailed facial features",
                "style": "realistic",
                "positive_prompt": "realistic portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "Modern portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality": {
                "description": "Modern photography style with natural lighting and professional camera",
                "style": "realistic",
                "positive_prompt": "modern portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "Studio photography, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality": {
                "description": "Studio photography with natural lighting and professional camera quality",
                "style": "realistic",
                "positive_prompt": "studio photography, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "Professional headshot, natural lighting, studio background, sharp focus, high resolution, detailed facial features": {
                "description": "Corporate headshot style with natural lighting and high resolution",
                "style": "realistic",
                "positive_prompt": "professional headshot, natural lighting, studio background, sharp focus, high resolution, detailed facial features",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "High fashion portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality": {
                "description": "High fashion magazine style with 4K quality and detailed features",
                "style": "realistic",
                "positive_prompt": "high fashion portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "Celebrity style portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful": {
                "description": "Celebrity magazine style with professional photography",
                "style": "realistic",
                "positive_prompt": "celebrity style portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "Magazine cover portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality": {
                "description": "Magazine cover style with professional camera and sharp focus",
                "style": "realistic",
                "positive_prompt": "magazine cover portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "Artistic portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality": {
                "description": "Artistic photography with natural skin texture and detailed features",
                "style": "realistic",
                "positive_prompt": "artistic portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "Cinematic portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality": {
                "description": "Cinematic movie style with dramatic lighting and professional quality",
                "style": "realistic",
                "positive_prompt": "cinematic portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            "Vintage portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful": {
                "description": "Vintage photography style with classic studio lighting",
                "style": "realistic",
                "positive_prompt": "vintage portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
                "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed",
            },
            # Cartoon/Animation themes
            "Van Gogh style": {
                "description": "Post-impressionist painting style with bold colors and expressive brushstrokes",
                "style": "cartoon",
                "positive_prompt": "Van Gogh style painting, post-impressionist, bold colors, expressive brushstrokes, artistic",
                "negative_prompt": "realistic, photograph, modern, clean, smooth",
            },
            "Cartoon version": {
                "description": "Classic cartoon style with bold outlines and vibrant colors",
                "style": "cartoon",
                "positive_prompt": "cartoon style, bold outlines, vibrant colors, animated, stylized",
                "negative_prompt": "realistic, photograph, detailed, natural",
            },
            "Anime style": {
                "description": "Japanese anime style with large eyes and stylized features",
                "style": "cartoon",
                "positive_prompt": "anime style, Japanese animation, large eyes, stylized features, vibrant colors",
                "negative_prompt": "realistic, photograph, natural, detailed",
            },
        }
        return theme_descriptions.get(
            theme_name,
            {
                "description": "Default theme",
                "style": "realistic",
                "positive_prompt": theme_name,
                "negative_prompt": "",
            },
        )

    def detect_theme_style(theme_name):
        return "realistic"  # Default to realistic

    def get_enhanced_prompts_for_theme(theme_name, sketch_type="face", user_prompt=""):
        """
        Enhanced prompt generation that PRIORITIZES user prompt first to avoid truncation.
        This ensures user details like hair color are not cut off.
        """
        theme_info = get_theme_info(theme_name)

        if user_prompt:
            # PRIORITY: User prompt comes FIRST to avoid truncation
            if sketch_type == "face":
                if theme_info["style"] == "realistic":
                    enhanced_prompt = f"{user_prompt}, realistic portrait of one person, {theme_info['positive_prompt']}, detailed facial features, proper anatomy, single face"
                    enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error"
                else:  # cartoon
                    enhanced_prompt = f"{user_prompt}, anime style portrait, {theme_info['positive_prompt']}, stylized features, clean lines, detailed illustration, single face"
                    enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, realistic, photograph"
            else:  # full_body
                if theme_info["style"] == "realistic":
                    enhanced_prompt = f"{user_prompt}, realistic full body portrait of one person, {theme_info['positive_prompt']}, proper anatomy, single figure, natural pose"
                    enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people"
                else:  # cartoon
                    enhanced_prompt = f"{user_prompt}, anime style full body portrait, {theme_info['positive_prompt']}, stylized features, clean lines, detailed illustration, single figure, dynamic pose"
                    enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people, realistic, photograph"
        else:
            # Fallback: No user prompt, use theme-based construction (original logic)
            if sketch_type == "face":
                if theme_info["style"] == "realistic":
                    enhanced_prompt = f"realistic portrait of one person, {theme_info['positive_prompt']}, detailed facial features, proper anatomy, single face"
                    enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error"
                else:  # cartoon
                    enhanced_prompt = f"anime style portrait, {theme_info['positive_prompt']}, stylized features, clean lines, detailed illustration, single face"
                    enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, realistic, photograph"
            else:  # full_body
                if theme_info["style"] == "realistic":
                    enhanced_prompt = f"realistic full body portrait of one person, {theme_info['positive_prompt']}, proper anatomy, single figure, natural pose"
                    enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people"
                else:  # cartoon
                    enhanced_prompt = f"anime style full body portrait, {theme_info['positive_prompt']}, stylized features, clean lines, detailed illustration, single figure, dynamic pose"
                    enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people, realistic, photograph"

        return enhanced_prompt, enhanced_negative, theme_info.get("style", "realistic")

    THEME_CATEGORIES = ["Realistic", "Cartoon/Animation", "Artistic"]

# ======================================================================
# CENTRALIZED PARAMETER SYSTEM - Single source of truth for all parameters
# ======================================================================
# This ensures automatic sync across all sections - change here, updates everywhere!
PRESET_PARAMETERS = {
    "realistic": {
        "face": {
            "Default": {
                "controlnet_conditioning_scale": 0.95,
                "guidance_scale": 6.5,
                "num_inference_steps": 30,
                "control_type": "scribble",
            },
            "Balanced": {
                "controlnet_conditioning_scale": 0.9,
                "guidance_scale": 6.0,
                "num_inference_steps": 30,
                "control_type": "scribble",
            },
            "Creative": {
                "controlnet_conditioning_scale": 0.8,
                "guidance_scale": 7.0,
                "num_inference_steps": 30,
                "control_type": "scribble",
            },
        },
        "full_body": {
            "Default": {
                "controlnet_conditioning_scale": 0.60,
                "guidance_scale": 8.0,
                "num_inference_steps": 30,
                "control_type": "scribble",
            },
            "Balanced": {
                "controlnet_conditioning_scale": 0.70,
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
                "control_type": "scribble",
            },
            "Creative": {
                "controlnet_conditioning_scale": 0.8,
                "guidance_scale": 7.0,
                "num_inference_steps": 30,
                "control_type": "scribble",
            },
        },
    },
    "cartoon": {
        "face": {
            "Default": {
                "controlnet_conditioning_scale": 0.95,
                "guidance_scale": 7.0,
                "num_inference_steps": 32,
                "control_type": "scribble",
            },
            "Balanced": {
                "controlnet_conditioning_scale": 0.9,
                "guidance_scale": 6.5,
                "num_inference_steps": 34,
                "control_type": "scribble",
            },
            "Creative": {
                "controlnet_conditioning_scale": 0.8,
                "guidance_scale": 6.0,
                "num_inference_steps": 36,
                "control_type": "scribble",
            },
        },
        "full_body": {
            "Default": {
                "controlnet_conditioning_scale": 0.95,
                "guidance_scale": 7.0,
                "num_inference_steps": 32,
                "control_type": "scribble",
            },
            "Balanced": {
                "controlnet_conditioning_scale": 0.9,
                "guidance_scale": 6.5,
                "num_inference_steps": 34,
                "control_type": "scribble",
            },
            "Creative": {
                "controlnet_conditioning_scale": 0.8,
                "guidance_scale": 6.0,
                "num_inference_steps": 36,
                "control_type": "scribble",
            },
        },
    },
    "ultra_realistic": {
        "face": {
            "Default": {
                "controlnet_conditioning_scale": 0.95,
                "guidance_scale": 7.0,
                "num_inference_steps": 32,
                "control_type": "canny",
            },
            "Balanced": {
                "controlnet_conditioning_scale": 0.9,
                "guidance_scale": 6.5,
                "num_inference_steps": 34,
                "control_type": "canny",
            },
            "Creative": {
                "controlnet_conditioning_scale": 0.58,
                "guidance_scale": 3.0,
                "num_inference_steps": 36,
                "control_type": "canny",
            },
        },
        "full_body": {
            "Default": {
                "controlnet_conditioning_scale": 0.95,
                "guidance_scale": 7.0,
                "num_inference_steps": 32,
                "control_type": "canny",
            },
            "Balanced": {
                "controlnet_conditioning_scale": 0.4,
                "guidance_scale": 8.0,
                "num_inference_steps": 34,
                "control_type": "canny",
            },
            "Creative": {
                "controlnet_conditioning_scale": 0.4,
                "guidance_scale": 8.0,
                "num_inference_steps": 36,
                "control_type": "canny",
            },
        },
    },
}


def get_parameters_from_preset(art_style, sketch_type, preset_choice):
    """
    Get parameters from the centralized PRESET_PARAMETERS dictionary.
    This ensures automatic sync across all sections.
    """
    # Extract preset key from the full preset choice string
    if "Creative" in preset_choice:
        preset_key = "Creative"
    elif "Balanced" in preset_choice:
        preset_key = "Balanced"
    else:
        preset_key = "Default"

    try:
        # Get parameters from the centralized dictionary
        params_dict = PRESET_PARAMETERS[art_style][sketch_type][preset_key]

        # Separate control_type from other parameters
        control_type = params_dict["control_type"]
        optimal_params = {k: v for k, v in params_dict.items() if k != "control_type"}

        return optimal_params, control_type
    except KeyError as e:
        # Fallback to default realistic face parameters if something goes wrong
        st.error(
            f"⚠️ Parameter lookup error: {e}. Using default realistic face parameters."
        )
        fallback_dict = PRESET_PARAMETERS["realistic"]["face"]["Default"]
        control_type = fallback_dict["control_type"]
        optimal_params = {k: v for k, v in fallback_dict.items() if k != "control_type"}
        return optimal_params, control_type


try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass  # HEIC/HEIF support not available


# --- Hugging Face API Integration ---
def call_hf_sdxl_api(image_bytes, prompt, negative_prompt=""):
    """
    SDXL API with Hugging Face token authentication
    """

    # Get Hugging Face token from environment or use default
    hf_token = os.getenv("HF_TOKEN", "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")

    if not hf_token or hf_token == "your_token_here":
        st.error("❌ No Hugging Face token found")
        st.info("💡 Set your token with: export HF_TOKEN=your_token_here")
        st.info("💡 Get your token from: https://huggingface.co/settings/tokens")
        st.info("💡 Switching to local ControlNet...")

        # Fallback to local ControlNet
        return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)

    # Try HTTP SDXL API first
    result = call_sdxl_http_api(image_bytes, prompt, negative_prompt, hf_token)

    if result:
        st.success("✅ SDXL HTTP API generation successful!")
        return result

    # Try Gradio SDXL API
    result = call_sdxl_gradio_api(image_bytes, prompt, negative_prompt, hf_token)

    if result:
        st.success("✅ SDXL Gradio API generation successful!")
        return result

    # If all SDXL APIs fail, try local ControlNet
    st.warning("⚠️ All SDXL APIs failed")
    st.info("💡 Switching to local ControlNet for similar quality...")
    return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)


def call_sdxl_http_api(image_bytes, prompt, negative_prompt, hf_token):
    """
    Call SDXL using HTTP API with token authentication
    """
    try:
        # Use regular SDXL API but with enhanced prompts that include sketch information
        sdxl_endpoints = [
            {
                "name": "SDXL Base 1.0",
                "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
                "headers": {"Authorization": f"Bearer {hf_token}"},
            },
            {
                "name": "SD 1.5 (Fallback)",
                "url": "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
                "headers": {"Authorization": f"Bearer {hf_token}"},
            },
        ]

        for endpoint in sdxl_endpoints:
            try:
                # Enhanced prompt that includes sketch guidance
                enhanced_prompt = f"professional photography, {prompt}, based on sketch outline, following the sketch composition, high quality, detailed, natural lighting, sharp focus, beautiful, 8k, ultra detailed"

                # BALANCED negative prompt - much less aggressive
                balanced_negative = "deformed, distorted, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, floating limbs, mutated hands and fingers, out of focus, long neck, long body, mutated hands and fingers, missing arms, missing legs, extra arms, extra legs, malformed limbs, mutated limbs, double image, blurred, ugly, disgusting, amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed, mutated, anatomical, anatomical error, goofy, unrealistic, cartoon, anime, painting, drawing, sketch, illustration, low quality, blurry, pixelated"

                # Use user's negative prompt if provided, otherwise use balanced one
                final_negative = (
                    negative_prompt if negative_prompt else balanced_negative
                )

                # Prepare payload
                payload = {
                    "inputs": enhanced_prompt,
                    "parameters": {
                        "negative_prompt": final_negative,
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                    },
                }

                # Make request
                response = requests.post(
                    endpoint["url"],
                    headers=endpoint["headers"],
                    json=payload,
                    timeout=120,
                )

                if response.status_code == 200:
                    # Handle different response formats
                    if response.headers.get("content-type", "").startswith("image/"):
                        # Direct image response
                        result_image = Image.open(io.BytesIO(response.content))
                        return result_image
                    else:
                        # JSON response with base64 image
                        try:
                            import base64

                            image_data = response.json()
                            if "image" in image_data:
                                image_bytes = base64.b64decode(image_data["image"])
                                result_image = Image.open(io.BytesIO(image_bytes))
                                return result_image
                        except:
                            pass

                elif response.status_code == 401:
                    st.warning(
                        f"❌ {endpoint['name']}: Unauthorized - check your token"
                    )
                elif response.status_code == 503:
                    st.warning(
                        f"❌ {endpoint['name']}: Service unavailable - model is loading"
                    )
                else:
                    st.warning(f"❌ {endpoint['name']}: HTTP {response.status_code}")

            except Exception as e:
                st.warning(f"❌ {endpoint['name']} failed: {e}")
                continue

        return None

    except Exception as e:
        st.error(f"❌ HTTP API error: {e}")
        return None


def call_sdxl_gradio_api(image_bytes, prompt, negative_prompt, hf_token):
    """
    Call SDXL using Gradio Client with token authentication
    """
    try:
        import tempfile
        import os
        from gradio_client import Client

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(image_bytes)
            temp_image_path = tmp_file.name

        try:
            # Try the sketch-to-3d space which actually processes sketches
            client = Client("linoyts/sketch-to-3d", hf_token=hf_token)

            # Generate image using the sketch
            result = client.predict(
                image={
                    "background": temp_image_path,
                    "layers": [temp_image_path],
                    "composite": temp_image_path,
                    "id": None,
                },
                prompt=f"professional photography, {prompt}, based on sketch outline, following the sketch composition, high quality, detailed, natural lighting, sharp focus, beautiful, 8k, ultra detailed",
                negative_prompt=negative_prompt
                or "deformed, distorted, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, floating limbs, mutated hands and fingers, out of focus, long neck, long body, mutated hands and fingers, missing arms, missing legs, extra arms, extra legs, malformed limbs, mutated limbs, double image, blurred, ugly, disgusting, amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed, mutated, anatomical, anatomical error, goofy, unrealistic, cartoon, anime, painting, drawing, sketch, illustration, low quality, blurry, pixelated",
                style_name="Photographic",
                num_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                api_name="/preprocess_image",
            )

            if result and len(result) > 0:
                processed_image_path = result[0]

                if os.path.exists(processed_image_path):
                    result_image = Image.open(processed_image_path)
                    return result_image

        except Exception as e:
            st.warning(f"Gradio client error: {e}")
            return None

    except ImportError:
        st.error("❌ gradio_client not installed")
        st.info("💡 Install with: pip install gradio_client")
        return None
    finally:
        # Clean up temp file
        if "temp_image_path" in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass


def call_local_controlnet_fallback(image_bytes, prompt, negative_prompt):
    """
    Fallback to local ControlNet backend
    """
    try:
        # Get backend URL from global configuration
        backend_url = get_backend_url()

        # Check if using default URL
        if "your-ngrok-url-here" in backend_url:
            st.warning("⚠️ Backend URL not configured")
            st.info("💡 Update ngrok_config.py with your current ngrok URL")
            st.info("   Edit CURRENT_NGROK_URL in ngrok_config.py")
            return None

        files = {"image": ("input.png", image_bytes, "image/png")}
        data = {"prompt": prompt, "negative_prompt": negative_prompt}

        response = requests.post(backend_url, files=files, data=data, timeout=120)

        if response.status_code == 200:
            result_image = bytes_to_image(response.content)
            st.success("✅ Local ControlNet generation successful!")
            return result_image
        else:
            st.error(f"❌ Local backend error: {response.status_code}")
            return None

    except Exception as local_error:
        st.error(f"❌ Local backend also failed: {local_error}")
        st.info("💡 Please check your Colab backend is running")
        return None


def call_gradio_sdxl_robust(image_bytes, prompt, negative_prompt, endpoint):
    """
    Call SDXL using Gradio Client with better error handling
    """
    try:
        import tempfile
        import os
        from gradio_client import Client

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(image_bytes)
            temp_image_path = tmp_file.name

        try:
            # Initialize client
            client = Client(endpoint["space"])

            # Generate image
            result = client.predict(
                image={
                    "background": temp_image_path,
                    "layers": [temp_image_path],
                    "composite": temp_image_path,
                    "id": None,
                },
                prompt=prompt,
                negative_prompt=negative_prompt
                or "deformed, distorted, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, floating limbs, mutated hands and fingers, out of focus, long neck, long body, mutated hands and fingers, missing arms, missing legs, extra arms, extra legs, malformed limbs, mutated limbs, double image, blurred, ugly, disgusting, amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed, mutated, anatomical, anatomical error, goofy, unrealistic, cartoon, anime, painting, drawing, sketch, illustration, low quality, blurry, pixelated",
                style_name=endpoint["style"],
                num_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                api_name="/preprocess_image",
            )

            if result and len(result) > 0:
                processed_image_path = result[0]

                if os.path.exists(processed_image_path):
                    result_image = Image.open(processed_image_path)
                    return result_image

        except Exception as e:
            print(f"Gradio client error: {e}")
            return None

    except ImportError:
        st.error("❌ gradio_client not installed")
        st.info("💡 Install with: pip install gradio_client")
        return None
    finally:
        # Clean up temp file
        if "temp_image_path" in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass


def call_3d_model_api(image_bytes, prompt, negative_prompt=""):
    """
    Call Hugging Face 3D Model API using Gradio Client
    """
    try:
        import tempfile
        import os
        from gradio_client import Client

        # Create a temporary file for the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(image_bytes)
            temp_image_path = tmp_file.name

        try:
            # Initialize the client
            client = Client("linoyts/sketch-to-3d")

            # Stage 1: Preprocess the image
            result = client.predict(
                image={
                    "background": temp_image_path,
                    "layers": [temp_image_path],
                    "composite": temp_image_path,
                    "id": None,
                },
                prompt=prompt,
                negative_prompt=negative_prompt
                or "deformed, distorted, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, floating limbs, mutated hands and fingers, out of focus, long neck, long body, mutated hands and fingers, missing arms, missing legs, extra arms, extra legs, malformed limbs, mutated limbs, double image, blurred, ugly, disgusting, amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed, mutated, anatomical, anatomical error, goofy, unrealistic, cartoon, anime, painting, drawing, sketch, illustration, low quality, blurry, pixelated",
                style_name="3D Model",
                num_steps=8,
                guidance_scale=5,
                controlnet_conditioning_scale=0.85,
                api_name="/preprocess_image",
            )

            if result and len(result) > 0:
                processed_image_path = result[0]  # Get the processed image path

                # Stage 2: Generate 3D model
                result_3d = client.predict(
                    image=processed_image_path,
                    multiimages=[],
                    seed=0,
                    ss_guidance_strength=7.5,
                    ss_sampling_steps=12,
                    slat_guidance_strength=3,
                    slat_sampling_steps=12,
                    multiimage_algo="stochastic",
                    api_name="/image_to_3d",
                )

                if result_3d and "video" in result_3d:
                    st.success("✅ 3D model generation successful!")
                    return {
                        "type": "3d",
                        "video_path": result_3d["video"],
                        "processed_image_path": processed_image_path,
                    }
                else:
                    st.error("❌ No 3D data in response")
                    return None
            else:
                st.error("❌ No processed image data in response")
                return None

        except Exception as e:
            st.warning("⚠️ 3D generation is currently unavailable")
            st.info("💡 Switching to local ControlNet for image generation...")

            # Fallback to local ControlNet for image generation
            try:
                # Use the same image and prompts with local backend
                # Get backend URL from global configuration
                backend_url = get_backend_url()

                files = {"image": ("input.png", image_bytes, "image/png")}
                data = {"prompt": prompt, "negative_prompt": negative_prompt}

                response = requests.post(
                    backend_url, files=files, data=data, timeout=120
                )

                if response.status_code == 200:
                    result_image = bytes_to_image(response.content)
                    st.success("✅ Local ControlNet generation successful!")
                    return {"type": "image", "image": result_image}
                else:
                    st.error(f"❌ Local backend error: {response.status_code}")
                    return None

            except Exception as local_error:
                st.error(f"❌ Local backend also failed: {local_error}")
                st.info("💡 Please check your Colab backend is running")
                return None
    finally:
        # Clean up temporary file
        if "temp_image_path" in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass


def call_kaggle_sdxl_api(image_bytes, prompt, negative_prompt=""):
    """
    Call Kaggle SDXL ControlNet API (highest quality)
    """
    try:
        # Get Kaggle backend URL from global configuration
        kaggle_url = get_backend_url()

        if "your-ngrok-url-here" in kaggle_url:
            st.warning("⚠️ Kaggle backend URL not configured")
            st.info("💡 Update ngrok_config.py with your current ngrok URL")
            st.info("💡 Switching to local ControlNet...")
            return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)

        # Enhanced prompt for SDXL quality
        enhanced_prompt = f"professional photography, {prompt}, high quality, detailed, natural lighting, sharp focus, beautiful, masterpiece, 8k, ultra detailed"

        # BALANCED negative prompt for SDXL
        balanced_negative = "deformed, distorted, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, floating limbs, mutated hands and fingers, out of focus, long neck, long body, mutated hands and fingers, missing arms, missing legs, extra arms, extra legs, malformed limbs, mutated limbs, double image, blurred, ugly, disgusting, amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed, mutated, anatomical, anatomical error, goofy, unrealistic, cartoon, anime, painting, drawing, sketch, illustration, low quality, blurry, pixelated"

        # Use user's negative prompt if provided, otherwise use balanced one
        final_negative = negative_prompt if negative_prompt else balanced_negative

        # Convert image to base64 for JSON request
        import base64

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare JSON request
        json_data = {
            "prompt": enhanced_prompt,
            "negative_prompt": final_negative,
            "image_data": image_base64,
            "control_type": "scribble",
            "num_inference_steps": 30,
        }

        # Add retry logic for 404 errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Show progress indicator
                with st.spinner(
                    "🎨 Generating image with Kaggle SDXL... This may take 3-5 minutes for high-quality results."
                ):
                    response = requests.post(
                        kaggle_url, json=json_data, timeout=600
                    )  # Extended timeout for SDXL

                if response.status_code == 200:
                    # Parse JSON response
                    result_json = response.json()

                    # Check if the response contains image data
                    if "image" in result_json:
                        # Decode base64 image data
                        result_image_bytes = base64.b64decode(result_json["image"])
                        result_image = bytes_to_image(result_image_bytes)
                        st.success("✅ Kaggle SDXL generation successful!")
                        return result_image
                    else:
                        st.error("❌ Invalid response format from backend")
                        st.info("💡 Switching to local ControlNet...")
                        return call_local_controlnet_fallback(
                            image_bytes, prompt, negative_prompt
                        )
                elif response.status_code == 404 and attempt < max_retries - 1:
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    st.error(f"❌ Kaggle backend error: {response.status_code}")
                    if response.status_code == 500:
                        st.error(f"Backend error details: {response.text}")
                    st.info("💡 Switching to local ControlNet...")
                    return call_local_controlnet_fallback(
                        image_bytes, prompt, negative_prompt
                    )

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.error("❌ All attempts timed out")
                    st.info("💡 Switching to local ControlNet...")
                    return call_local_controlnet_fallback(
                        image_bytes, prompt, negative_prompt
                    )

    except Exception as e:
        st.warning("⚠️ Kaggle SDXL is currently unavailable")
        st.error(f"Error details: {str(e)}")
        st.info("💡 Switching to local ControlNet for similar quality...")
        return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)


def call_controlnet_union_official_api(image_bytes, prompt, negative_prompt=""):
    """
    Call official ControlNet++ Union API (based on GitHub source)
    """
    try:
        # Get official ControlNet++ Union backend URL from global configuration
        union_url = get_backend_url()

        if "your-ngrok-url-here" in union_url:
            st.warning("⚠️ Official ControlNet++ Union backend URL not configured")
            st.info("💡 Update ngrok_config.py with your current ngrok URL")
            st.info("💡 Switching to local ControlNet...")
            return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)

        # Official ControlNet++ Union control types (from GitHub source)
        # Default to scribble (Type 2) for best sketch adherence
        control_type = "scribble"  # Can be: scribble, hed, canny, lineart, etc.

        # Convert image to base64 for JSON request
        import base64

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare JSON request with official parameters
        json_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_data": image_base64,
            "control_type": control_type,  # Official control type
            "num_inference_steps": 30,
        }

        # Add retry logic for 404 errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Show progress indicator
                with st.spinner(
                    "🎨 Generating image with Official ControlNet++ Union... This may take 3-5 minutes for high-quality results."
                ):
                    response = requests.post(union_url, json=json_data, timeout=600)

                if response.status_code == 200:
                    # Parse JSON response
                    result_json = response.json()

                    # Check if the response contains image data
                    if "image" in result_json:
                        # Decode base64 image data
                        result_image_bytes = base64.b64decode(result_json["image"])
                        result_image = bytes_to_image(result_image_bytes)
                        st.success(
                            "✅ Official ControlNet++ Union generation successful!"
                        )
                        return result_image
                    else:
                        st.error("❌ Invalid response format from backend")
                        st.info("💡 Switching to local ControlNet...")
                        return call_local_controlnet_fallback(
                            image_bytes, prompt, negative_prompt
                        )

                elif response.status_code == 404 and attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.error(
                        f"❌ Official ControlNet++ Union backend error: {response.status_code}"
                    )
                    if response.status_code == 500:
                        st.error(f"Backend error details: {response.text}")
                    st.info("💡 Switching to local ControlNet...")
                    return call_local_controlnet_fallback(
                        image_bytes, prompt, negative_prompt
                    )

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.error("❌ All attempts timed out")
                    st.info("💡 Switching to local ControlNet...")
                    return call_local_controlnet_fallback(
                        image_bytes, prompt, negative_prompt
                    )

    except Exception as e:
        st.warning("⚠️ Official ControlNet++ Union is currently unavailable")
        st.error(f"Error details: {str(e)}")
        st.info("💡 Switching to local ControlNet for similar quality...")
        return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)


def call_enhanced_controlnet_api(
    image_bytes,
    prompt,
    negative_prompt="",
    sketch_type="face",
    control_type="scribble",
    art_style="realistic",
    preset_choice="Default (Recommended)",
):
    """
    Enhanced ControlNet++ Union API with optimal parameters from comprehensive testing
    Now supports different art styles (realistic vs cartoon/animation vs ultra_realistic)
    """
    try:
        # Get backend URL from global configuration
        backend_url = get_backend_url()

        if "your-ngrok-url-here" in backend_url:
            st.warning("⚠️ Backend URL not configured")
            st.info("💡 Update ngrok_config.py with your current ngrok URL")
            return None

        # Convert image to base64 for JSON request
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Choose parameter preset based on style and sketch detail level
        # Defaults are what worked best in your observations
        control_guidance_start = 0.0
        control_guidance_end = 0.8
        use_refiner = False

        if art_style == "ultra_realistic":
            # SDXL + Canny ControlNet
            use_refiner = True
            if (
                preset_choice.startswith("Default")
                or preset_choice == "Detailed (Current)"
            ):
                optimal_params = {
                    "controlnet_conditioning_scale": 0.7,
                    "guidance_scale": 3.5,
                    "num_inference_steps": 30,
                }
                control_guidance_end = 0.8
            elif "Balanced" in preset_choice:
                # For simple sketches
                optimal_params = {
                    "controlnet_conditioning_scale": 0.65,
                    "guidance_scale": 3.2,
                    "num_inference_steps": 32,
                }
                control_guidance_end = 0.65
            else:  # Creative / Very simple
                optimal_params = {
                    "controlnet_conditioning_scale": 0.58,
                    "guidance_scale": 3.0,
                    "num_inference_steps": 36,
                }
                control_guidance_end = 0.5
        else:
            # Use centralized parameter system for automatic sync
            optimal_params, control_type = get_parameters_from_preset(
                art_style, sketch_type, preset_choice
            )

        # ENHANCED PROMPT ENGINEERING FOR DIFFERENT STYLES
        if art_style == "realistic":
            if sketch_type == "face":
                enhanced_prompt = f"realistic photographic portrait, natural skin texture, professional studio lighting, shallow depth of field, sharp focus, single person, detailed eyes, natural hair, smooth facial geometry, no outline overlay, {prompt}"
                enhanced_negative = (
                    "cartoon, anime, illustration, drawing, painting, 3d render, cgi, doll-like, plastic skin, overprocessed, smooth flat shading, line overlay, heavy outlines, deformed, extra limbs, bad anatomy, multiple faces, blurry, low quality"
                    if not negative_prompt
                    else f"cartoon, anime, illustration, drawing, painting, 3d render, cgi, doll-like, plastic skin, overprocessed, smooth flat shading, line overlay, heavy outlines, deformed, extra limbs, bad anatomy, multiple faces, blurry, low quality, {negative_prompt}"
                )
            else:
                enhanced_prompt = f"{prompt}, realistic full body photo, professional photography, studio or clean backdrop, correct human anatomy, natural pose, single person, detailed clothing folds, smooth body geometry, no outline overlay"
                enhanced_negative = (
                    "cartoon, anime, illustration, drawing, painting, 3d render, cgi, doll-like, plastic skin, overprocessed, smooth flat shading, line overlay, heavy outlines, deformed, extra limbs, bad anatomy, multiple people, blurry, low quality"
                    if not negative_prompt
                    else f"cartoon, anime, illustration, drawing, painting, 3d render, cgi, doll-like, plastic skin, overprocessed, smooth flat shading, line overlay, heavy outlines, deformed, extra limbs, bad anatomy, multiple people, blurry, low quality, {negative_prompt}"
                )
        elif art_style == "ultra_realistic":
            if sketch_type == "face":
                # Ultra realistic face prompts with sketch detail preservation
                enhanced_prompt = f"photorealistic portrait, DSLR photography, natural skin texture with pores, detailed facial features, professional studio lighting, sharp focus, beautiful, proper anatomy, ultra high resolution, 8k quality, HDR, bokeh, natural expression, realistic hair strands, detailed eyes and eyebrows, natural skin tone, single person, single face, no group, no crowd, {prompt}"
                enhanced_negative = (
                    "deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered"
                    if not negative_prompt
                    else f"deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered, {negative_prompt}"
                )
            else:  # full_body
                # Ultra realistic full body prompts with sketch detail preservation
                enhanced_prompt = f"{prompt}, photorealistic full body portrait, DSLR photography, natural skin texture with pores, detailed facial features, wearing casual clothing, decent attire, appropriate dress, proper anatomy, single figure, natural pose, ultra high resolution, 8k quality, HDR, bokeh, natural expression, realistic hair strands, detailed eyes and eyebrows, natural skin tone, single person, no group, no crowd"
                enhanced_negative = (
                    "deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered"
                    if not negative_prompt
                    else f"deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered, {negative_prompt}"
                )
        else:  # cartoon/animation
            if sketch_type == "face":
                # Cartoon face prompts
                enhanced_prompt = f"anime style portrait, vibrant colors, stylized features, clean lines, detailed illustration, single face, {prompt}"
                enhanced_negative = (
                    "deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, realistic, photograph"
                    if not negative_prompt
                    else f"deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, realistic, photograph, {negative_prompt}"
                )
            else:  # full_body
                # Cartoon full body prompts
                enhanced_prompt = f"anime style full body portrait, vibrant colors, stylized features, clean lines, detailed illustration, single figure, dynamic pose, {prompt}"
                enhanced_negative = (
                    "deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people, realistic, photograph"
                    if not negative_prompt
                    else f"deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people, realistic, photograph, {negative_prompt}"
                )

        # Prepare enhanced JSON request with optimal parameters
        json_data = {
            "prompt": enhanced_prompt,
            "negative_prompt": enhanced_negative,
            "image_data": image_base64,
            "control_type": control_type,
            "sketch_type": sketch_type,
            "art_style": art_style,
            "num_inference_steps": optimal_params["num_inference_steps"],
            "controlnet_conditioning_scale": optimal_params[
                "controlnet_conditioning_scale"
            ],
            "guidance_scale": optimal_params["guidance_scale"],
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "use_refiner": use_refiner,
        }

        # Make the API call with retry logic
        start_time = time.time()

        # Add retry logic for connection issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with st.spinner(
                    f"🎨 Generating {sketch_type.replace('_', ' ')} {art_style} painting... (Attempt {attempt + 1}/{max_retries})"
                ):
                    response = requests.post(
                        f"{backend_url}/generate",
                        json=json_data,
                        timeout=600,  # Extended timeout for all SDXL modes
                    )
                # If we get here, the request was successful
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    st.warning(f"⏰ Attempt {attempt + 1} timed out. Retrying...")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    st.error(
                        "⏰ All attempts timed out. The backend is processing a complex image."
                    )
                    st.info(
                        "💡 Try reducing image complexity or using a simpler preset."
                    )
                    st.stop()

            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    st.warning(
                        f"🔌 Attempt {attempt + 1} failed due to connection error. Retrying..."
                    )
                    time.sleep(3)  # Wait longer for connection issues
                    continue
                else:
                    st.error(
                        "🔌 Connection error after all retries. Please check if the backend is running."
                    )
                    st.info(
                        "💡 Ensure your ngrok URL is up to date and backend is accessible."
                    )
                    st.stop()

            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(2)
                    continue
                else:
                    st.error(f"❌ All attempts failed with error: {str(e)}")
                    st.stop()

        try:
            generation_time = time.time() - start_time
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    # Convert base64 image back to PIL
                    image_data = base64.b64decode(result["image"])
                    result_image = Image.open(io.BytesIO(image_data))

                    # Show success message with parameters used
                    if art_style == "realistic" and "Creative" in preset_choice:
                        st.success(
                            f"✅ {sketch_type.title()} realistic generation successful!"
                        )
                        st.info(f"🎯 Using Scribble ControlNet for enhanced creativity")
                    else:
                        st.success(
                            f"✅ {sketch_type.title()} {art_style} generation successful!"
                        )

                    st.info(
                        f"📊 Used optimal parameters: ControlNet={optimal_params['controlnet_conditioning_scale']}, Guidance={optimal_params['guidance_scale']}"
                    )
                    st.info(f"⏱️ Generation time: {generation_time:.1f} seconds")

                    # Add download button
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Download Image",
                        data=buf.getvalue(),
                        file_name=f"{art_style}_{sketch_type}_{int(time.time())}.png",
                        mime="image/png",
                    )

                    # Display the result
                    st.image(
                        result_image,
                        caption=f"{art_style} {sketch_type} Result",
                        use_container_width=True,
                    )
                    return result_image
                else:
                    st.error(
                        f"❌ Backend error: {result.get('detail', 'Unknown error')}"
                    )
                    return None
            else:
                st.error(f"❌ Backend error: {response.status_code}")
                st.error(f"Response: {response.text}")
                return None
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection error. Please check if the backend is running.")
            st.info("💡 Ensure your ngrok URL is up to date.")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None
    except Exception as e:
        st.error(f"❌ Function error: {str(e)}")
        return None


def call_linoyts_approach_api(image_bytes, prompt, negative_prompt=""):
    """
    Call Linoyts Approach API (exact linoyts/sketch-to-3d implementation)
    """
    try:
        # Get Linoyts Approach backend URL from global configuration
        linoyts_url = get_backend_url()

        if "your-ngrok-url-here" in linoyts_url:
            st.warning("⚠️ Linoyts Approach backend URL not configured")
            st.info("💡 Update ngrok_config.py with your current ngrok URL")
            st.info("💡 Switching to local ControlNet...")
            return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)

        # Linoyts approach: Simple, effective prompts (no manual gender detection)
        enhanced_prompt = f"realistic human portrait, {prompt}, detailed face, natural proportions, studio lighting, high quality, professional photography"

        # Linoyts approach: Simple negative prompt
        balanced_negative = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blurry, distorted, disfigured, multiple faces, multiple heads"

        # Use user's negative prompt if provided, otherwise use balanced one
        final_negative = negative_prompt if negative_prompt else balanced_negative

        # Convert image to base64 for JSON request
        import base64

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare JSON request
        json_data = {
            "prompt": enhanced_prompt,
            "negative_prompt": final_negative,
            "image_data": image_base64,
            "control_type": "scribble",
            "num_inference_steps": 25,  # Linoyts optimal value
        }

        # Add retry logic for 404 errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Show progress indicator
                with st.spinner(
                    "🎨 Generating image with Linoyts Approach... This may take 3-5 minutes for high-quality results."
                ):
                    response = requests.post(linoyts_url, json=json_data, timeout=600)

                if response.status_code == 200:
                    # Parse JSON response
                    result_json = response.json()

                    # Check if the response contains image data
                    if "image" in result_json:
                        # Decode base64 image data
                        result_image_bytes = base64.b64decode(result_json["image"])
                        result_image = bytes_to_image(result_image_bytes)
                        st.success("✅ Linoyts Approach generation successful!")
                        return result_image
                    else:
                        st.error("❌ Invalid response format from backend")
                        st.info("💡 Switching to local ControlNet...")
                        return call_local_controlnet_fallback(
                            image_bytes, prompt, negative_prompt
                        )

                elif response.status_code == 404 and attempt < max_retries - 1:
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    st.error(
                        f"❌ Linoyts Approach backend error: {response.status_code}"
                    )
                    if response.status_code == 500:
                        st.error(f"Backend error details: {response.text}")
                    st.info("💡 Switching to local ControlNet...")
                    return call_local_controlnet_fallback(
                        image_bytes, prompt, negative_prompt
                    )

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.error("❌ All attempts timed out")
                    st.info("💡 Switching to local ControlNet...")
                    return call_local_controlnet_fallback(
                        image_bytes, prompt, negative_prompt
                    )
            except Exception as inner_e:
                if attempt < max_retries - 1:
                    st.warning(
                        f"⚠️ Attempt {attempt + 1} failed: {str(inner_e)}. Retrying..."
                    )
                    time.sleep(2)
                    continue
                else:
                    st.error(f"❌ All attempts failed with error: {str(inner_e)}")
                    st.info("💡 Switching to local ControlNet...")
                    return call_local_controlnet_fallback(
                        image_bytes, prompt, negative_prompt
                    )

        # If we get here, all attempts failed
        st.warning("⚠️ Linoyts Approach is currently unavailable")
        st.info("💡 Switching to local ControlNet for similar quality...")
        return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)
    except Exception as e:
        st.warning("⚠️ Linoyts Approach is currently unavailable")
        st.info("💡 Switching to local ControlNet for similar quality...")
        return call_local_controlnet_fallback(image_bytes, prompt, negative_prompt)


# --- Model options ---
MODEL_OPTIONS = [
    "Pix2Pix",
    "ControlNet",
    "FLUX",
    "SDXL ControlNet (HF)",
    "SDXL ControlNet (Kaggle)",
    "SDXL + ControlNet (Kaggle)",
    "SDXL + ControlNet Official (Kaggle)",
    "Linoyts Approach (Kaggle)",
    "3D Model (HF)",
]

# --- Enhanced theme options ---
# Organize themes by subject type
THEME_CATEGORIES = {
    "Human": [
        # ControlNet Portrait Themes (Roughly.app style)
        "Realistic portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "Professional headshot, natural lighting, studio background, sharp focus, high resolution, detailed facial features",
        "Studio photography, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "High fashion portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality",
        "Celebrity style portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "Magazine cover portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "Artistic portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality",
        "Cinematic portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "Vintage portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "Modern portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        # Original themes
        "Van Gogh style",
        "Cartoon version",
        "Turn into a cyborg",
        "Watercolor painting",
        "Anime style",
        "Futuristic robot",
        "Old photo",
        "Pop art",
        "Impressionist painting",
        "Disney animation style",
        "Studio Ghibli style",
        "Pixar 3D animation",
        "Realistic humanoid",
        "Cartoon Network style",
        "Marvel comic style",
        "DC Comics style",
        "Japanese manga",
        "European oil painting",
        "Digital art",
        "Steampunk style",
        "Cyberpunk aesthetic",
        "Fantasy art",
        "Sci-fi concept art",
        "Horror movie style",
        "Romantic painting",
        # TV show/movie themes
        "FRIENDS sitcom style",
        "Twilight Saga cinematic",
        "Stranger Things retro horror",
        "Seinfeld 90s sitcom",
        "Game of Thrones fantasy",
        "Breaking Bad drama",
        "The Office mockumentary",
        "Star Wars sci-fi",
        "Marvel superhero",
        "Disney princess",
        "Harry Potter magic",
        "Lord of the Rings epic",
        "Rick and Morty surreal",
        "Simpsons cartoon",
        "SpongeBob underwater",
        "Avatar: The Last Airbender",
        "Narcos crime drama",
        "Peaky Blinders vintage",
        "Squid Game suspense",
        # Famous AI generator styles
        "DALL-E surrealism",
        "Midjourney photorealism",
        "Stable Diffusion fantasy",
        "DreamBooth custom style",
        "Deep Dream psychedelic",
        "RunwayML cinematic",
        "NightCafe creative",
        "Artbreeder blend",
    ],
    "Architecture": [
        "Modern skyscraper",
        "Classic European landmark",
        "Futuristic cityscape",
        "Ancient temple",
        "Famous bridge",
        "Interior design",
        "Street scene",
        "Nature landscape",
    ],
    "Object": [
        "Furniture design",
        "Product rendering",
        "Vehicle concept art",
        "Gadget prototype",
        "Fashion design sketch",
        "Jewelry design",
        "Sports equipment",
        "Musical instrument",
        "Book cover art",
        "Poster design",
        "Still life painting",
        "Food photography",
        "Animal illustration",
    ],
    "Animal": [
        "Animal illustration",
        "Nature landscape",
        "Cartoon version",
        "Anime style",
        "Pixar 3D animation",
        "Studio Ghibli style",
        "Disney animation style",
        "Realistic humanoid",
    ],
    "Other": [
        "DALL-E surrealism",
        "Midjourney photorealism",
        "Stable Diffusion fantasy",
        "DreamBooth custom style",
        "Deep Dream psychedelic",
        "RunwayML cinematic",
        "NightCafe creative",
        "Artbreeder blend",
    ],
}

SUBJECT_TYPES = list(THEME_CATEGORIES.keys())

st.set_page_config(page_title="AI Drawing to Painting Studio", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Inter:wght@400;700&display=swap');

html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    min-height: 100vh;
    font-family: 'Inter', 'Montserrat', 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(120deg, #232256 0%, #6a11cb 20%, #2575fc 50%, #43e97b 80%, #38f9d7 100%);
    background-attachment: fixed;
}

body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 70% 30%, #43e97b 0%, #38f9d7 40%, transparent 80%),
                radial-gradient(circle at 20% 80%, #6a11cb 0%, #232256 50%, transparent 100%),
                radial-gradient(circle at 80% 80%, #2575fc 0%, #43e97b 60%, transparent 100%);
    opacity: 0.22;
    z-index: 0;
    pointer-events: none;
    animation: livelyBg 18s ease-in-out infinite alternate;
}
@keyframes livelyBg {
    0% {
        background-position: 70% 30%, 20% 80%, 80% 80%;
        opacity: 0.22;
    }
    100% {
        background-position: 60% 40%, 30% 70%, 70% 90%;
        opacity: 0.32;
    }
}

.main .block-container {
    background: #fff;
    border-radius: 32px;
    box-shadow: 0 8px 32px rgba(127,127,213,0.13), 0 1.5px 0 0 #b993f8;
    padding: 40px 32px 32px 32px;
    margin-top: 32px;
    margin-bottom: 32px;
    max-width: 900px;
    border: 1.5px solid #b993f8;
    position: relative;
    z-index: 2;
}

.stApp { background: transparent; }

.sidebar .sidebar-content {
    background: rgba(255,255,255,0.92);
    border-radius: 24px;
    box-shadow: 0 4px 16px rgba(127,127,213,0.07);
    padding: 24px 16px;
    border: 1.5px solid #b993f8;
    backdrop-filter: blur(10px);
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 50%, #43e97b 100%);
    color: #222;
    border: none;
    border-radius: 16px;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.7em 2em;
    box-shadow: 0 2px 8px rgba(127,127,213,0.10);
    transition: background 0.2s, transform 0.1s;
    margin: 0.5em 0;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(90deg, #43e97b 0%, #2575fc 50%, #6a11cb 100%);
    color: #fff;
    transform: scale(1.04);
    box-shadow: 0 4px 16px rgba(127,127,213,0.13);
}

.stTextInput > div > input, .stSelectbox > div > div, .stSlider > div {
    border-radius: 12px !important;
    border: 1.5px solid #b993f8 !important;
    background: #fff !important;
    font-size: 1.05rem;
    color: #222;
    box-shadow: 0 1px 4px rgba(127,127,213,0.04);
}

.stTextInput > div > input:focus, .stSelectbox > div > div:focus-within {
    border: 2px solid #2575fc !important;
    outline: none !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Montserrat', 'Inter', sans-serif;
    color: #2d2d2d;
    letter-spacing: 1px;
    text-shadow: 0 2px 8px #b993f833;
    margin-bottom: 0.2em;
}

.stCaption, .stMarkdown p {
    color: #222;
    font-size: 1.08rem;
}

/* PURE WHITE SKETCH AREA */
.st-drawable-canvas {
    background: #fff !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 24px #b993f822 !important;
    border: 2px solid #b993f833 !important;
    z-index: 2 !important;
    transform: none !important;
    scale: 1 !important;
    max-width: none !important;
    width: 800px !important;
    height: 600px !important;
}

::-webkit-scrollbar {
    width: 10px;
    background: #e0e7ef;
}
::-webkit-scrollbar-thumb {
    background: #b993f8;
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Prepare art supplies PNG as base64 for HTML embedding
art_supplies_path = os.path.join(os.path.dirname(__file__), "art_supplies.png")
with open(art_supplies_path, "rb") as f:
    art_supplies_b64 = base64.b64encode(f.read()).decode("utf-8")
art_supplies_url = f"data:image/png;base64,{art_supplies_b64}"

# Place this at the very top of the script, before any Streamlit UI
st.markdown(
    f"""
<div class="background-art-animated">
    <img src="{art_supplies_url}" class="background-art-img" />
</div>
<style>
.background-art-animated {{
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}}
.background-art-img {{
    position: absolute;
    top: 0;
    left: -20vw;
    width: 140vw;
    min-width: 1200px;
    opacity: 0.13;
    filter: blur(0.2px);
    animation: artImgFlow 9.5s ease-in-out infinite alternate;
}}
@keyframes artImgFlow {{
    0%   {{ left: -20vw; }}
    100% {{ left: -5vw; }}
}}
.main .block-container {{
    background: rgba(255,255,255,0.98);
    border-radius: 32px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.10), 0 1.5px 0 0 #FFD700;
    padding: 40px 32px 32px 32px;
    margin-top: 32px;
    margin-bottom: 32px;
    max-width: 900px;
    border: 2px solid #f3e7c3;
    position: relative;
    z-index: 2;
}}
.st-drawable-canvas {{
    background: #fff !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 24px #FFD70022 !important;
    border: 2px solid #FFD70033 !important;
    z-index: 2 !important;
    transform: none !important;
    scale: 1 !important;
    max-width: none !important;
    width: 800px !important;
    height: 600px !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# Main title and description
st.title("🎨 Drawing to Painting App")
st.markdown(
    "Transform your sketches into stunning paintings using AI-powered generation!"
)

# Smart AI Button - Top Right Corner
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("")  # Spacer for title

with col2:
    st.markdown("")  # Spacer

with col3:
    # Smart AI Button - Expandable
    if st.button(
        "🤖 Smart AI", key="smart_ai_toggle", type="primary", use_container_width=True
    ):
        st.session_state.smart_ai_expanded = not st.session_state.get(
            "smart_ai_expanded", False
        )

# Smart AI Interface (Expandable)
if st.session_state.get("smart_ai_expanded", False):
    st.markdown("---")
    st.subheader("🤖 Smart AI Analysis")

    # Create two columns for mode selection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🚀 Auto Mode**")
        st.markdown("AI analyzes your sketch and automatically selects everything!")
        auto_mode = st.button(
            "🤖 Use Auto Mode", key="auto_mode_btn", use_container_width=True
        )

        if auto_mode:
            st.session_state.smart_mode = "auto"
            st.success(
                "✅ Auto Mode activated! AI will analyze your sketch automatically."
            )
            # Don't rerun to preserve canvas

    with col2:
        st.markdown("**🎯 Guided Mode**")
        st.markdown("AI uses your sidebar selections and generates perfect prompts!")
        guided_mode = st.button(
            "🎯 Use Guided Mode", key="guided_mode_btn", use_container_width=True
        )

        if guided_mode:
            st.session_state.smart_mode = "guided"
            st.success("✅ Guided Mode activated! AI will use your sidebar selections.")
            # Don't rerun to preserve canvas

    # Smart Mode Status and Analysis
    if st.session_state.get("smart_mode"):
        st.markdown("---")
        if st.session_state.smart_mode == "auto":
            st.info(
                "🤖 **Auto Mode Active**: AI will analyze your sketch and configure everything automatically!"
            )
        else:
            st.info(
                "🎯 **Guided Mode Active**: AI will use your sidebar selections and generate perfect prompts!"
            )

            # Show guided mode options
            if st.session_state.smart_mode == "guided":
                st.markdown("**🎯 Your Current Selections:**")

                # Custom subject type input for guided mode
                st.markdown("**👤 Customize Subject Type:**")
                custom_subject = st.text_input(
                    "Type custom subject (optional):",
                    placeholder="e.g., astronaut, superhero, robot, warrior princess",
                )
                if custom_subject.strip():
                    st.session_state.custom_subject = custom_subject.strip()
                    st.success(f"✅ Custom subject set: {custom_subject.strip()}")

                col1, col2 = st.columns(2)
                with col1:
                    st.info(
                        f"🎨 Theme: {st.session_state.get('current_theme', 'Not selected')}"
                    )
                    st.info(
                        f"📝 Sketch Type: {st.session_state.get('sketch_type', 'full_body').replace('_', ' ').title()}"
                    )
                with col2:
                    st.info(
                        f"🎭 Art Style: {st.session_state.get('art_style', 'realistic').title()}"
                    )
                    subject_display = st.session_state.get(
                        "custom_subject",
                        st.session_state.get("subject_type", "Not selected"),
                    )
                    st.info(f"👤 Subject Type: {subject_display.title()}")

        # Show smart mode controls
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset Smart Mode", key="reset_smart"):
                st.session_state.smart_mode = None
                st.session_state.auto_prompt = None
                st.session_state.guided_prompt = None
                st.session_state.auto_params = None
                # Don't rerun to preserve canvas

        with col2:
            if st.button("📋 Show AI Analysis", key="show_analysis"):
                if (
                    hasattr(st.session_state, "last_analysis")
                    and st.session_state.last_analysis
                ):
                    st.json(st.session_state.last_analysis)
                else:
                    st.warning(
                        "⚠️ No analysis available. Draw or upload a sketch first."
                    )

        # Show analysis trigger button when sketch is available
        if st.session_state.get("input_image") is not None:
            st.markdown("---")
            st.subheader("🔍 Ready to Analyze!")

            # Get the image for analysis (preserve both canvas and uploaded file)
            analysis_image = st.session_state.get("input_image")

            if st.session_state.smart_mode == "auto":
                # Auto mode: Analyze and configure everything
                st.markdown(
                    """
                <style>
                .robotic-terminal {
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    border: 2px solid #00d4ff;
                    border-radius: 15px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
                    position: relative;
                    overflow: hidden;
                }
                .robotic-terminal::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
                    animation: scan 2s infinite;
                }
                @keyframes scan {
                    0% { left: -100%; }
                    100% { left: 100%; }
                }
                .robotic-terminal .terminal-text {
                    color: #ffffff !important;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    margin: 0;
                    text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
                }
                .analyze-btn {
                    background: linear-gradient(45deg, #00d4ff, #0099cc);
                    border: none;
                    border-radius: 25px;
                    color: white;
                    padding: 15px 30px;
                    font-size: 16px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
                }
                .analyze-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.6);
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="robotic-terminal"><p class="terminal-text">🤖 AUTO MODE: AI NEURAL NETWORK READY</p><p class="terminal-text">📊 SKETCH ANALYSIS PROTOCOL INITIALIZED</p><p class="terminal-text">🎯 AUTOMATIC PARAMETER OPTIMIZATION ENABLED</p></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    "🔍 Analyze Sketch with AI", key="analyze_auto", type="primary"
                ):
                    # Ensure canvas data is ready for analysis
                    # Note: canvas_result is defined later, so we use session state
                    if st.session_state.get("canvas_data") is not None:
                        st.session_state.canvas_initialized = True
                        st.info("🎨 Canvas data ready for AI analysis")

                        # Set analyzing flag to show animation overlay
                        st.session_state.ai_analyzing = True

                        # Convert canvas data to image bytes for AI analysis
                        canvas_array = st.session_state.canvas_data.astype("uint8")
                        canvas_image = Image.fromarray(canvas_array)

                        # Convert to bytes
                        img_byte_arr = io.BytesIO()
                        canvas_image.save(img_byte_arr, format="PNG")
                        image_bytes = img_byte_arr.getvalue()

                        # Get sidebar selections
                        sidebar_selections = {
                            "theme": st.session_state.get("theme", "portrait"),
                            "subject_type": st.session_state.get(
                                "subject_type", "auto"
                            ),
                            "art_style": st.session_state.get("art_style", "realistic"),
                            "sketch_type": st.session_state.get(
                                "sketch_type", "full_body"
                            ),
                        }

                        # Call the AI analysis function
                        analysis = auto_generate_from_sketch(
                            image_bytes, sidebar_selections
                        )

                        # Reset analyzing flag
                        st.session_state.ai_analyzing = False

                        if analysis:
                            st.success("✅ AI analysis completed successfully!")
                        else:
                            st.error("❌ AI analysis failed.")
                    else:
                        st.warning("⚠️ Please draw something on the canvas first!")
                        st.session_state.ai_analyzing = False
                        st.stop()

                    # Show enhanced robotic terminal animation after clicking
                    st.markdown(
                        """
                    <style>
                    .terminal-animation {
                        background: #000000;
                        border: 2px solid #00ff00;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 20px 0;
                        font-family: 'Courier New', monospace;
                        color: #00ff00;
                        position: relative;
                        overflow: hidden;
                        line-height: 1.6;
                    }
                    .typing-line {
                        position: relative;
                        margin: 8px 0;
                        min-height: 20px;
                    }
                    .typing-text {
                        display: inline-block;
                        overflow: hidden;
                        white-space: nowrap;
                        border-right: 3px solid #00ff00;
                        animation: blink-caret 1s step-end infinite;
                        font-weight: bold;
                    }
                    @keyframes blink-caret {
                        from, to { border-color: transparent; }
                        50% { border-color: #00ff00; }
                    }
                    .cursor {
                        display: inline-block;
                        width: 8px;
                        height: 16px;
                        background: #00ff00;
                        margin-left: 2px;
                        animation: cursor-blink 1s infinite;
                        box-shadow: 0 0 10px #00ff00;
                    }
                    @keyframes cursor-blink {
                        0%, 50% { opacity: 1; }
                        51%, 100% { opacity: 0; }
                    }
                    .terminal-header {
                        color: #00ff00;
                        font-weight: bold;
                        margin-bottom: 15px;
                        border-bottom: 1px solid #00ff00;
                        padding-bottom: 10px;
                    }
                    .status-line {
                        color: #00ff00;
                        opacity: 0.8;
                        font-size: 12px;
                        margin-top: 5px;
                    }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Enhanced terminal animation container - print all lines at once
                    terminal_container = st.container()

                    with terminal_container:
                        # Terminal header
                        st.markdown(
                            '<div class="terminal-animation"><div class="terminal-header">🤖 AI NEURAL NETWORK TERMINAL v2.0</div></div>',
                            unsafe_allow_html=True,
                        )

                        # Enhanced CSS-based word-by-word typing animation
                        st.markdown(
                            """
                        <style>
                        .typing-animation {
                            overflow: hidden;
                            white-space: nowrap;
                            border-right: 3px solid #00ff00;
                            animation: typing 4s steps(40, end), blink-caret 0.75s step-end infinite;
                            font-family: 'Courier New', monospace;
                            color: #00ff00;
                            font-weight: bold;
                            text-shadow: 0 0 5px rgba(0, 255, 0, 0.5);
                            position: relative;
                        }
                        @keyframes typing {
                            from { width: 0; }
                            to { width: 100%; }
                        }
                        @keyframes blink-caret {
                            from, to { border-color: transparent; }
                            50% { border-color: #00ff00; }
                        }
                        .typing-animation::after {
                            content: '';
                            position: absolute;
                            right: -3px;
                            top: 0;
                            width: 3px;
                            height: 100%;
                            background: #00ff00;
                            animation: blink-caret 0.75s step-end infinite;
                            box-shadow: 0 0 10px #00ff00;
                        }
                        .typing-line-animated {
                            margin: 8px 0;
                            min-height: 20px;
                            background: rgba(0, 0, 0, 0.1);
                            padding: 5px 10px;
                            border-radius: 5px;
                        }
                        </style>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Animated typing lines with CSS animations
                        lines = [
                            "🚀 INITIALIZING NEURAL NETWORK...",
                            "📊 LOADING SKETCH ANALYSIS MODULES...",
                            "🔍 SCANNING SKETCH PATTERNS...",
                            "👤 DETECTING GENDER INDICATORS...",
                            "📐 ANALYZING OBJECT GEOMETRY...",
                            "🎨 GENERATING OPTIMIZED PROMPT...",
                            "✅ ANALYSIS COMPLETE! READY TO GENERATE...",
                        ]

                        for i, line in enumerate(lines):
                            # Each line appears with staggered delay (slower for better effect)
                            st.markdown(
                                f"""
                            <div class="terminal-animation">
                                <div class="typing-line-animated">
                                    <div class="typing-animation" style="animation-delay: {i * 1.2}s;">
                                        {line}
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Add natural delay between terminal windows
                            if i < len(lines) - 1:  # Don't delay after the last line
                                time.sleep(0.5)

                        # Status line
                        st.markdown(
                            '<div class="terminal-animation"><div class="status-line">Status: Ready | Mode: Auto | Protocol: Active</div></div>',
                            unsafe_allow_html=True,
                        )

                    with st.spinner("🤖 AI is analyzing your sketch..."):
                        try:
                            # Get canvas data and convert to image for analysis
                            canvas_data = st.session_state.get("canvas_data")
                            if canvas_data is None:
                                # Try to restore from backup first
                                if st.session_state.get("canvas_backup") is not None:
                                    canvas_data = st.session_state.get("canvas_backup")
                                    st.session_state.canvas_data = canvas_data
                                    st.session_state.canvas_initialized = True
                                    st.info("🔄 Restored sketch from backup")
                                # Try to get from input_image as fallback
                                elif st.session_state.get("input_image") is not None:
                                    img_source = st.session_state.get("input_image")
                                    img_buffer = io.BytesIO()
                                    img_source.save(img_buffer, format="PNG")
                                    img_bytes = img_buffer.getvalue()
                                    # Store this as canvas data for consistency
                                    st.session_state.canvas_data = np.array(img_source)
                                    st.session_state.canvas_initialized = True
                                else:
                                    st.error(
                                        "❌ No sketch available for analysis. Please draw or upload a sketch first."
                                    )
                                    st.stop()

                            # Now process the canvas data (either original, restored, or fallback)
                            if canvas_data is not None:
                                # Convert canvas data to PIL Image
                                canvas_array = canvas_data.astype("uint8")
                                img_source = Image.fromarray(canvas_array)

                                # Convert image to bytes for analysis
                                img_buffer = io.BytesIO()
                                img_source.save(img_buffer, format="PNG")
                                img_bytes = img_buffer.getvalue()

                                # Ensure canvas data is preserved in session state
                                st.session_state.canvas_data = canvas_data
                                st.session_state.canvas_initialized = True

                            # Get current sidebar selections including theme and subject_type
                            sidebar_selections = {
                                "theme": st.session_state.get(
                                    "current_theme", "portrait"
                                ),
                                "art_style": st.session_state.get(
                                    "art_style", "realistic"
                                ),
                                "sketch_type": st.session_state.get(
                                    "sketch_type", "full_body"
                                ),
                                "preset_choice": st.session_state.get(
                                    "preset_choice", "Default"
                                ),
                                "subject_type": st.session_state.get(
                                    "custom_subject",
                                    st.session_state.get("subject_type", "auto"),
                                ),
                            }

                            st.info(
                                f"🔍 Analyzing with: {sidebar_selections['art_style']} style, {sidebar_selections['sketch_type']} type, {sidebar_selections['theme']} theme"
                            )

                            # Run AI analysis
                            analysis = auto_generate_from_sketch(
                                img_bytes, sidebar_selections
                            )
                            if analysis:
                                # Reset analyzing flag when analysis is complete
                                st.session_state.ai_analyzing = False

                                st.session_state.last_analysis = analysis
                                # Set AI-generated prompt for auto-fill
                                if "generated_prompt" in analysis:
                                    st.session_state.ai_generated_prompt = analysis[
                                        "generated_prompt"
                                    ]
                                    # Auto-fill the prompt in the sidebar
                                    st.session_state.user_prompt = analysis[
                                        "generated_prompt"
                                    ]
                                    st.success(
                                        "🚀 AI analysis complete! Prompt auto-filled and auto-generating your painting..."
                                    )
                                    # Auto-trigger generation
                                    st.session_state.trigger_auto_generation = True
                                    st.session_state.auto_generate_now = True

                                    # Show analysis results in robotic terminal style
                                    st.markdown(
                                        """
                                    <style>
                                    .analysis-results-terminal {
                                        background: #000000;
                                        border: 2px solid #00ff00;
                                        border-radius: 10px;
                                        padding: 20px;
                                        margin: 20px 0;
                                        font-family: 'Courier New', monospace;
                                        color: #00ff00;
                                        position: relative;
                                        overflow: hidden;
                                        line-height: 1.6;
                                        max-height: 400px;
                                        overflow-y: auto;
                                    }
                                    .analysis-line {
                                        margin: 8px 0;
                                        padding: 5px 0;
                                        border-bottom: 1px solid #00ff00;
                                        opacity: 0;
                                        animation: fadeInLine 0.5s ease-in forwards;
                                    }
                                    .analysis-line:nth-child(1) { animation-delay: 0.1s; }
                                    .analysis-line:nth-child(2) { animation-delay: 0.2s; }
                                    .analysis-line:nth-child(3) { animation-delay: 0.3s; }
                                    .analysis-line:nth-child(4) { animation-delay: 0.4s; }
                                    .analysis-line:nth-child(5) { animation-delay: 0.5s; }
                                    .analysis-line:nth-child(6) { animation-delay: 0.6s; }
                                    .analysis-line:nth-child(7) { animation-delay: 0.7s; }
                                    .analysis-line:nth-child(8) { animation-delay: 0.8s; }
                                    .analysis-line:nth-child(9) { animation-delay: 0.9s; }
                                    .analysis-line:nth-child(10) { animation-delay: 1.0s; }
                                    @keyframes fadeInLine {
                                        from { opacity: 0; transform: translateX(-10px); }
                                        to { opacity: 1; transform: translateX(0); }
                                    }
                                    .analysis-key {
                                        color: #00ff00;
                                        font-weight: bold;
                                    }
                                    .analysis-value {
                                        color: #ffffff;
                                        margin-left: 10px;
                                    }
                                    .analysis-separator {
                                        color: #00ff00;
                                        margin: 0 5px;
                                    }
                                    </style>
                                    """,
                                        unsafe_allow_html=True,
                                    )

                                    # Create robotic terminal display for analysis results
                                    analysis_container = st.container()
                                    with analysis_container:
                                        st.markdown(
                                            '<div class="analysis-results-terminal">',
                                            unsafe_allow_html=True,
                                        )

                                        # Display analysis results line by line with robotic formatting
                                        if "subject" in analysis:
                                            st.markdown(
                                                f'<div class="analysis-line"><span class="analysis-key">🤖 SUBJECT:</span><span class="analysis-separator">></span><span class="analysis-value">{analysis["subject"]}</span></div>',
                                                unsafe_allow_html=True,
                                            )

                                        if "person_details" in analysis:
                                            person = analysis["person_details"]
                                            st.markdown(
                                                f'<div class="analysis-line"><span class="analysis-key">👤 PERSON DETAILS:</span><span class="analysis-separator">></span><span class="analysis-value">Loading...</span></div>',
                                                unsafe_allow_html=True,
                                            )
                                            time.sleep(0.3)
                                            st.markdown(
                                                f'<div class="analysis-line"><span class="analysis-key">   🚹 Gender:</span><span class="analysis-separator">></span><span class="analysis-value">{person.get("gender", "Unknown")}</span></div>',
                                                unsafe_allow_html=True,
                                            )
                                            time.sleep(0.2)
                                            st.markdown(
                                                f'<div class="analysis-line"><span class="analysis-key">   📅 Age:</span><span class="analysis-separator">></span><span class="analysis-value">{person.get("age_group", "Unknown")}</span></div>',
                                                unsafe_allow_html=True,
                                            )
                                            time.sleep(0.2)
                                            st.markdown(
                                                f'<div class="analysis-line"><span class="analysis-key">   🎭 Pose:</span><span class="analysis-separator">></span><span class="analysis-value">{person.get("pose", "Unknown")}</span></div>',
                                                unsafe_allow_html=True,
                                            )
                                            time.sleep(0.2)
                                            st.markdown(
                                                f'<div class="analysis-line"><span class="analysis-key">   👕 Clothing:</span><span class="analysis-separator">></span><span class="analysis-value">{person.get("clothing", "Unknown")}</span></div>',
                                                unsafe_allow_html=True,
                                            )

                                        if "generated_prompt" in analysis:
                                            st.markdown(
                                                f'<div class="analysis-line"><span class="analysis-key">🎨 GENERATED PROMPT:</span><span class="analysis-separator">></span><span class="analysis-value">{analysis["generated_prompt"][:100]}{"..." if len(analysis["generated_prompt"]) > 100 else ""}</span></div>',
                                                unsafe_allow_html=True,
                                            )

                                        if "recommendations" in analysis:
                                            st.markdown(
                                                f'<div class="analysis-line"><span class="analysis-key">💡 RECOMMENDATIONS:</span><span class="analysis-separator">></span><span class="analysis-value">{analysis["recommendations"][:100]}{"..." if len(analysis["recommendations"]) > 100 else ""}</span></div>',
                                                unsafe_allow_html=True,
                                            )

                                        st.markdown(
                                            '<div class="analysis-line"><span class="analysis-key">✅ ANALYSIS COMPLETE</span><span class="analysis-separator">></span><span class="analysis-value">Ready for generation!</span></div>',
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown("</div>", unsafe_allow_html=True)

                                        # Add relevant emojis and status
                                        st.markdown(
                                            """
                                        <div style="text-align: center; margin: 20px 0; padding: 15px; background: linear-gradient(135deg, #00ff00 0%, #00cc00 100%); border-radius: 10px; color: #000; font-weight: bold;">
                                            🤖 AI Analysis Complete! 🎨 Ready to Generate! 🚀
                                        </div>
                                        """,
                                            unsafe_allow_html=True,
                                        )

                                        # Add more lively emojis and robot-themed elements
                                        st.markdown(
                                            """
                                        <div style="text-align: center; margin: 15px 0; padding: 10px; background: rgba(0, 255, 0, 0.1); border-radius: 8px; border: 1px solid #00ff00;">
                                            <span style="font-size: 24px; margin: 0 5px;">🤖</span>
                                            <span style="font-size: 24px; margin: 0 5px;">⚡</span>
                                            <span style="font-size: 24px; margin: 0 5px;">🎯</span>
                                            <span style="font-size: 24px; margin: 0 5px;">🧠</span>
                                            <span style="font-size: 24px; margin: 0 5px;">🔬</span>
                                            <span style="font-size: 24px; margin: 0 5px;">💻</span>
                                        </div>
                                        """,
                                            unsafe_allow_html=True,
                                        )
                                else:
                                    st.warning(
                                        "⚠️ AI analysis completed but no prompt was generated."
                                    )
                            else:
                                st.error("❌ AI analysis failed. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Error during AI analysis: {str(e)}")
                            st.info(
                                "💡 Please check if all required modules are loaded and try again."
                            )

            elif st.session_state.smart_mode == "guided":
                # Guided mode: Analyze with user's sidebar selections
                st.markdown(
                    """
                <style>
                .robotic-terminal-guided {
                    background: linear-gradient(135deg, #2d1b69 0%, #11998e 50%, #38ef7d 100%);
                    border: 2px solid #38ef7d;
                    border-radius: 15px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 0 20px rgba(56, 239, 125, 0.3);
                    position: relative;
                    overflow: hidden;
                }
                .robotic-terminal-guided::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(56, 239, 125, 0.2), transparent);
                    animation: scan-guided 2s infinite;
                }
                @keyframes scan-guided {
                    0% { left: -100%; }
                    100% { left: 100%; }
                }
                .robotic-terminal-guided .terminal-text-guided {
                    color: #ffffff !important;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    margin: 0;
                    text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="robotic-terminal-guided"><p class="terminal-text-guided">🎯 GUIDED MODE: USER PREFERENCES LOADED</p><p class="terminal-text-guided">📋 CUSTOM PARAMETERS CONFIGURED</p><p class="terminal-text-guided">🤖 AI PROMPT GENERATION OPTIMIZED</p></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    "🔍 Analyze Sketch with AI", key="analyze_guided", type="primary"
                ):
                    # Ensure canvas data is ready for analysis
                    # Note: canvas_result is defined later, so we use session state
                    if st.session_state.get("canvas_data") is not None:
                        st.session_state.canvas_initialized = True
                        st.info("🎨 Canvas data ready for AI analysis")

                        # Set analyzing flag to show animation overlay
                        st.session_state.ai_analyzing = True

                        # Convert canvas data to image bytes for AI analysis
                        canvas_array = st.session_state.canvas_data.astype("uint8")
                        canvas_image = Image.fromarray(canvas_array)

                        # Convert to bytes
                        img_byte_arr = io.BytesIO()
                        canvas_image.save(img_byte_arr, format="PNG")
                        image_bytes = img_byte_arr.getvalue()

                        # Get sidebar selections
                        sidebar_selections = {
                            "theme": st.session_state.get("theme", "portrait"),
                            "subject_type": st.session_state.get(
                                "subject_type", "auto"
                            ),
                            "art_style": st.session_state.get("art_style", "realistic"),
                            "sketch_type": st.session_state.get(
                                "sketch_type", "full_body"
                            ),
                        }

                        # Call the AI analysis function
                        analysis = guided_generate_from_sketch(
                            image_bytes, sidebar_selections
                        )

                        # Reset analyzing flag
                        st.session_state.ai_analyzing = False

                        if analysis:
                            st.success("✅ AI analysis completed successfully!")
                        else:
                            st.error("❌ AI analysis failed.")
                    else:
                        st.warning("⚠️ Please draw something on the canvas first!")
                        st.session_state.ai_analyzing = False
                        st.stop()

                    # Show enhanced robotic terminal animation after clicking
                    st.markdown(
                        """
                    <style>
                    .terminal-animation-guided {
                        background: #000000;
                        border: 2px solid #38ef7d;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 20px 0;
                        font-family: 'Courier New', monospace;
                        color: #38ef7d;
                        position: relative;
                        overflow: hidden;
                        line-height: 1.6;
                    }
                    .typing-line-guided {
                        position: relative;
                        margin: 8px 0;
                        min-height: 20px;
                    }
                    .typing-text-guided {
                        display: inline-block;
                        overflow: hidden;
                        white-space: nowrap;
                        border-right: 3px solid #38ef7d;
                        animation: blink-caret-guided 1s step-end infinite;
                        font-weight: bold;
                    }
                    @keyframes blink-caret-guided {
                        from, to { border-color: transparent; }
                        50% { border-color: #38ef7d; }
                    }
                    .cursor-guided {
                        display: inline-block;
                        width: 8px;
                        height: 16px;
                        background: #38ef7d;
                        margin-left: 2px;
                        animation: cursor-blink-guided 1s infinite;
                        box-shadow: 0 0 10px #38ef7d;
                    }
                    @keyframes cursor-blink-guided {
                        0%, 50% { opacity: 1; }
                        51%, 100% { opacity: 0; }
                    }
                    .terminal-header-guided {
                        color: #38ef7d;
                        font-weight: bold;
                        margin-bottom: 15px;
                        border-bottom: 1px solid #38ef7d;
                        padding-bottom: 10px;
                    }
                    .status-line-guided {
                        color: #38ef7d;
                        opacity: 0.8;
                        font-size: 12px;
                        margin-top: 5px;
                    }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Enhanced terminal animation container - print all lines at once
                    terminal_container_guided = st.container()

                    with terminal_container_guided:
                        # Terminal header
                        st.markdown(
                            '<div class="terminal-animation-guided"><div class="terminal-header-guided">🎯 GUIDED MODE TERMINAL v2.0</div></div>',
                            unsafe_allow_html=True,
                        )

                        # Enhanced CSS-based word-by-word typing animation for guided mode
                        st.markdown(
                            """
                        <style>
                        .typing-animation-guided {
                            overflow: hidden;
                            white-space: nowrap;
                            border-right: 3px solid #38ef7d;
                            animation: typing-guided 4s steps(40, end), blink-caret-guided 0.75s step-end infinite;
                            font-family: 'Courier New', monospace;
                            color: #38ef7d;
                            font-weight: bold;
                            text-shadow: 0 0 5px rgba(56, 239, 125, 0.5);
                            position: relative;
                        }
                        @keyframes typing-guided {
                            from { width: 0; }
                            to { width: 100%; }
                        }
                        @keyframes blink-caret-guided {
                            from, to { border-color: transparent; }
                            50% { border-color: #38ef7d; }
                        }
                        .typing-animation-guided::after {
                            content: '';
                            position: absolute;
                            right: -3px;
                            top: 0;
                            width: 3px;
                            height: 100%;
                            background: #38ef7d;
                            animation: blink-caret-guided 0.75s step-end infinite;
                            box-shadow: 0 0 10px #38ef7d;
                        }
                        .typing-line-animated-guided {
                            margin: 8px 0;
                            min-height: 20px;
                            background: rgba(56, 239, 125, 0.1);
                            padding: 5px 10px;
                            border-radius: 5px;
                        }
                        </style>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Animated typing lines with CSS animations
                        lines_guided = [
                            "⚙️ LOADING USER PREFERENCES...",
                            "🔧 APPLYING CUSTOM PARAMETERS...",
                            "🎨 CONFIGURING ART STYLE: REALISTIC...",
                            "🔍 ANALYZING WITH GUIDED SETTINGS...",
                            "💡 GENERATING PERSONALIZED PROMPT...",
                            "✅ ANALYSIS COMPLETE! READY TO GENERATE...",
                        ]

                        for i, line in enumerate(lines_guided):
                            # Each line appears with staggered delay (slower for better effect)
                            st.markdown(
                                f"""
                            <div class="terminal-animation-guided">
                                <div class="typing-line-animated-guided">
                                    <div class="typing-animation-guided" style="animation-delay: {i * 1.2}s;">
                                        {line}
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Add natural delay between terminal windows
                            if (
                                i < len(lines_guided) - 1
                            ):  # Don't delay after the last line
                                time.sleep(0.5)

                        # Status line
                        st.markdown(
                            '<div class="terminal-animation-guided"><div class="status-line-guided">Status: Ready | Mode: Guided | Protocol: Active</div></div>',
                            unsafe_allow_html=True,
                        )

                    with st.spinner(
                        "🤖 AI is analyzing your sketch with your selections..."
                    ):
                        try:
                            # Get canvas data and convert to image for analysis
                            canvas_data = st.session_state.get("canvas_data")
                            if canvas_data is None:
                                # Try to restore from backup first
                                if st.session_state.get("canvas_backup") is not None:
                                    canvas_data = st.session_state.get("canvas_backup")
                                    st.session_state.canvas_data = canvas_data
                                    st.session_state.canvas_initialized = True
                                    st.info("🔄 Restored sketch from backup")
                                # Try to get from input_image as fallback
                                elif st.session_state.get("input_image") is not None:
                                    img_source = st.session_state.get("input_image")
                                    img_buffer = io.BytesIO()
                                    img_source.save(img_buffer, format="PNG")
                                    img_bytes = img_buffer.getvalue()
                                    # Store this as canvas data for consistency
                                    st.session_state.canvas_data = np.array(img_source)
                                    st.session_state.canvas_initialized = True
                                else:
                                    st.error(
                                        "❌ No sketch available for analysis. Please draw or upload a sketch first."
                                    )
                                    st.stop()

                            # Now process the canvas data (either original, restored, or fallback)
                            if canvas_data is not None:
                                # Convert canvas data to PIL Image
                                canvas_array = canvas_data.astype("uint8")
                                img_source = Image.fromarray(canvas_array)

                                # Convert image to bytes for analysis
                                img_buffer = io.BytesIO()
                                img_source.save(img_buffer, format="PNG")
                                img_bytes = img_buffer.getvalue()

                                # Ensure canvas data is preserved in session state
                                st.session_state.canvas_data = canvas_data
                                st.session_state.canvas_initialized = True

                            # Get current sidebar selections including theme and subject_type
                            sidebar_selections = {
                                "theme": st.session_state.get(
                                    "current_theme", "portrait"
                                ),
                                "art_style": st.session_state.get(
                                    "art_style", "realistic"
                                ),
                                "sketch_type": st.session_state.get(
                                    "sketch_type", "full_body"
                                ),
                                "preset_choice": st.session_state.get(
                                    "preset_choice", "Default"
                                ),
                                "subject_type": st.session_state.get(
                                    "custom_subject",
                                    st.session_state.get("subject_type", "auto"),
                                ),
                            }

                            st.info(
                                f"🔍 Analyzing with: {sidebar_selections['art_style']} style, {sidebar_selections['sketch_type']} type, {sidebar_selections['theme']} theme"
                            )

                            # Run AI analysis with user's guided selections
                            analysis = guided_generate_from_sketch(
                                img_bytes, sidebar_selections
                            )
                            if analysis:
                                # Reset analyzing flag when analysis is complete
                                st.session_state.ai_analyzing = False

                                st.session_state.last_analysis = analysis
                                # Set AI-generated prompt for auto-fill
                                if "generated_prompt" in analysis:
                                    st.session_state.guided_prompt = analysis[
                                        "generated_prompt"
                                    ]
                                    # Auto-fill the prompt in the sidebar
                                    st.session_state.user_prompt = analysis[
                                        "generated_prompt"
                                    ]
                                    st.success(
                                        "🚀 AI analysis complete! Prompt auto-filled and auto-generating your painting..."
                                    )
                                    # Auto-trigger generation
                                    st.session_state.trigger_auto_generation = True
                                    st.session_state.auto_generate_now = True

                                    # Show analysis results in robotic terminal style
                                    st.markdown(
                                        """
                                    <style>
                                    .analysis-results-terminal-guided {
                                        background: #000000;
                                        border: 2px solid #38ef7d;
                                        border-radius: 10px;
                                        padding: 20px;
                                        margin: 20px 0;
                                        font-family: 'Courier New', monospace;
                                        color: #38ef7d;
                                        position: relative;
                                        overflow: hidden;
                                        line-height: 1.6;
                                        max-height: 400px;
                                        overflow-y: auto;
                                    }
                                    .analysis-line-guided {
                                        margin: 8px 0;
                                        padding: 5px 0;
                                        border-bottom: 1px solid #38ef7d;
                                        opacity: 0;
                                        animation: fadeInLineGuided 0.5s ease-in forwards;
                                    }
                                    .analysis-line-guided:nth-child(1) { animation-delay: 0.1s; }
                                    .analysis-line-guided:nth-child(2) { animation-delay: 0.2s; }
                                    .analysis-line-guided:nth-child(3) { animation-delay: 0.3s; }
                                    .analysis-line-guided:nth-child(4) { animation-delay: 0.4s; }
                                    .analysis-line-guided:nth-child(5) { animation-delay: 0.5s; }
                                    .analysis-line-guided:nth-child(6) { animation-delay: 0.6s; }
                                    .analysis-line-guided:nth-child(7) { animation-delay: 0.7s; }
                                    .analysis-line-guided:nth-child(8) { animation-delay: 0.8s; }
                                    .analysis-line-guided:nth-child(9) { animation-delay: 0.9s; }
                                    .analysis-line-guided:nth-child(10) { animation-delay: 1.0s; }
                                    @keyframes fadeInLineGuided {
                                        from { opacity: 0; transform: translateX(-10px); }
                                        to { opacity: 1; transform: translateX(0); }
                                    }
                                    .analysis-key-guided {
                                        color: #38ef7d;
                                        font-weight: bold;
                                    }
                                    .analysis-value-guided {
                                        color: #ffffff;
                                        margin-left: 10px;
                                    }
                                    .analysis-separator-guided {
                                        color: #38ef7d;
                                        margin: 0 5px;
                                    }
                                    </style>
                                    """,
                                        unsafe_allow_html=True,
                                    )

                                    # Create robotic terminal display for analysis results
                                    analysis_container_guided = st.container()
                                    with analysis_container_guided:
                                        st.markdown(
                                            '<div class="analysis-results-terminal-guided">',
                                            unsafe_allow_html=True,
                                        )

                                        # Display analysis results line by line with robotic formatting
                                        if "subject" in analysis:
                                            st.markdown(
                                                f'<div class="analysis-line-guided"><span class="analysis-key-guided">🎯 SUBJECT:</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">{analysis["subject"]}</span></div>',
                                                unsafe_allow_html=True,
                                            )

                                        if "person_details" in analysis:
                                            person = analysis["person_details"]
                                            st.markdown(
                                                f'<div class="analysis-line-guided"><span class="analysis-key-guided">👤 PERSON DETAILS:</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">Loading...</span></div>',
                                                unsafe_allow_html=True,
                                            )
                                            time.sleep(0.3)
                                            st.markdown(
                                                f'<div class="analysis-line-guided"><span class="analysis-key-guided">   🚹 Gender:</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">{person.get("gender", "Unknown")}</span></div>',
                                                unsafe_allow_html=True,
                                            )
                                            time.sleep(0.2)
                                            st.markdown(
                                                f'<div class="analysis-line-guided"><span class="analysis-key-guided">   📅 Age:</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">{person.get("age_group", "Unknown")}</span></div>',
                                                unsafe_allow_html=True,
                                            )
                                            time.sleep(0.2)
                                            st.markdown(
                                                f'<div class="analysis-line-guided"><span class="analysis-key-guided">   🎭 Pose:</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">{person.get("pose", "Unknown")}</span></div>',
                                                unsafe_allow_html=True,
                                            )
                                            time.sleep(0.2)
                                            st.markdown(
                                                f'<div class="analysis-line-guided"><span class="analysis-key-guided">   👕 Clothing:</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">{person.get("clothing", "Unknown")}</span></div>',
                                                unsafe_allow_html=True,
                                            )

                                        if "generated_prompt" in analysis:
                                            st.markdown(
                                                f'<div class="analysis-line-guided"><span class="analysis-key-guided">🎨 GENERATED PROMPT:</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">{analysis["generated_prompt"][:100]}{"..." if len(analysis["generated_prompt"]) > 100 else ""}</span></div>',
                                                unsafe_allow_html=True,
                                            )

                                        if "recommendations" in analysis:
                                            st.markdown(
                                                f'<div class="analysis-line-guided"><span class="analysis-key-guided">💡 RECOMMENDATIONS:</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">{analysis["recommendations"][:100]}{"..." if len(analysis["recommendations"]) > 100 else ""}</span></div>',
                                                unsafe_allow_html=True,
                                            )

                                        st.markdown(
                                            '<div class="analysis-line-guided"><span class="analysis-key-guided">✅ ANALYSIS COMPLETE</span><span class="analysis-separator-guided">></span><span class="analysis-value-guided">Ready for generation!</span></div>',
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown("</div>", unsafe_allow_html=True)

                                        # Add relevant emojis and status
                                        st.markdown(
                                            """
                                        <div style="text-align: center; margin: 20px 0; padding: 15px; background: linear-gradient(135deg, #38ef7d 0%, #00cc66 100%); border-radius: 10px; color: #000; font-weight: bold;">
                                            🎯 Guided Analysis Complete! 🎨 Ready to Generate! 🚀
                                        </div>
                                        """,
                                            unsafe_allow_html=True,
                                        )

                                        # Add more lively emojis and robot-themed elements
                                        st.markdown(
                                            """
                                        <div style="text-align: center; margin: 15px 0; padding: 10px; background: rgba(56, 239, 125, 0.1); border-radius: 8px; border: 1px solid #38ef7d;">
                                            <span style="font-size: 24px; margin: 0 5px;">🎯</span>
                                            <span style="font-size: 24px; margin: 0 5px;">⚡</span>
                                            <span style="font-size: 24px; margin: 0 5px;">🤖</span>
                                            <span style="font-size: 24px; margin: 0 5px;">🧠</span>
                                            <span style="font-size: 24px; margin: 0 5px;">🔬</span>
                                            <span style="font-size: 24px; margin: 0 5px;">💻</span>
                                        </div>
                                        """,
                                            unsafe_allow_html=True,
                                        )
                                else:
                                    st.warning(
                                        "⚠️ AI analysis completed but no prompt was generated."
                                    )
                            else:
                                # Reset analyzing flag on failure
                                st.session_state.ai_analyzing = False
                                st.error("❌ AI analysis failed. Please try again.")
                        except Exception as e:
                            # Reset analyzing flag on error
                            st.session_state.ai_analyzing = False
                            st.error(f"❌ Error during AI analysis: {str(e)}")
                            st.info(
                                "💡 Please check if all required modules are loaded and try again."
                            )
        else:
            st.info("💡 Draw or upload a sketch to enable AI analysis!")

st.markdown("---")

st.sidebar.header("1. Choose Subject & Model")
subject_type = st.sidebar.selectbox("Subject Type:", SUBJECT_TYPES, index=0)
model = st.sidebar.selectbox("AI Model:", MODEL_OPTIONS)

# Show helpful info for SDXL option
if model == "SDXL ControlNet (HF)":
    st.sidebar.info(
        "🎨 **SDXL ControlNet (HF)**: Professional quality using Hugging Face. No local GPU required!"
    )
elif model == "SDXL ControlNet (Kaggle)":
    st.sidebar.info(
        "🎨 **SDXL ControlNet (Kaggle)**: Maximum quality using Kaggle's powerful GPU. Best results!"
    )
elif model == "SDXL + ControlNet (Kaggle)":
    st.sidebar.info(
        "🎨 **SDXL + ControlNet (Kaggle)**: SDXL + Multiple ControlNets! Canny for ultra-realistic, Scribble for sketches and stickman drawings!"
    )
elif model == "SDXL + ControlNet Official (Kaggle)":
    st.sidebar.info(
        "🎨 **SDXL + ControlNet Official (Kaggle)**: SDXL + ControlNet backend with Scribble support for stickman to human conversion!"
    )
elif model == "Linoyts Approach (Kaggle)":
    st.sidebar.info(
        "🎨 **Linoyts Approach (Kaggle)**: Exact linoyts/sketch-to-3d implementation! Minimal preprocessing preserves sketch details!"
    )
elif model == "3D Model (HF)":
    st.sidebar.info(
        "🎨 **3D Model (HF)**: Generate 3D models from sketches using Hugging Face. Advanced AI!"
    )

# Show only relevant themes for the selected subject type
theme = st.sidebar.selectbox("Theme:", THEME_CATEGORIES[subject_type])

# Store theme and subject_type in session state for Smart AI access
st.session_state.current_theme = theme
st.session_state.subject_type = subject_type

# Show theme description and detected style
if theme:
    theme_info = get_theme_info(theme)
    detected_style = detect_theme_style(theme)

    # Display theme information
    st.sidebar.markdown(f"**Description:** {theme_info['description']}")
    st.sidebar.markdown(f"**Style:** {detected_style.title()}")

    # Show style indicator
    if detected_style == "realistic":
        st.sidebar.success("🎭 Realistic Style")
    else:
        st.sidebar.success("🎨 Cartoon/Animation Style")

# Smart AI prompt integration
default_prompt = ""
if st.session_state.get("ai_generated_prompt"):
    default_prompt = st.session_state.ai_generated_prompt
    st.sidebar.success(
        f"🤖 AI Generated Prompt: {default_prompt[:50]}..."
        if len(default_prompt) > 50
        else f"🤖 AI Generated Prompt: {default_prompt}"
    )
elif st.session_state.get("guided_prompt"):
    default_prompt = st.session_state.guided_prompt
    st.sidebar.success(
        f"🤖 AI Guided Prompt: {default_prompt[:50]}..."
        if len(default_prompt) > 50
        else f"🤖 AI Guided Prompt: {default_prompt}"
    )

# Preserve user prompt across reruns when AI modes are active
prompt = st.sidebar.text_input(
    "Describe your edit (optional):",
    value=st.session_state.get("user_prompt", default_prompt),
)
st.session_state.user_prompt = prompt

# Auto-generation trigger
if st.session_state.get("auto_generate_now") and st.session_state.get("user_prompt"):
    st.sidebar.success("🚀 Auto-generating painting with AI prompt...")
    # Reset the flag
    st.session_state.auto_generate_now = False
    # The generation will happen in the main logic below

st.sidebar.header("2. Drawing Tools")
stroke_color = st.sidebar.color_picker("Stroke color", "#000000")
stroke_width = st.sidebar.slider("Stroke width", 1, 20, 6)
background_color = st.sidebar.color_picker("Background color", "#ffffff")
fill_color = st.sidebar.color_picker("Fill color (for fill tool)", "#FFFFFF")
drawing_mode = st.sidebar.selectbox(
    "Tool",
    ("freedraw", "line", "rect", "circle", "transform", "point", "eraser", "fill"),
)

st.sidebar.header("3. Or Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image:", type=["png", "jpg", "jpeg"]
)

st.subheader("Draw your sketch or upload an image:")

# Main drawing canvas
canvas_width = 800
canvas_height = 600

# Initialize canvas data in session state if not exists
if "canvas_data" not in st.session_state:
    st.session_state.canvas_data = None
if "canvas_initialized" not in st.session_state:
    st.session_state.canvas_initialized = False
if "canvas_backup" not in st.session_state:
    st.session_state.canvas_backup = None
if "ai_analyzing" not in st.session_state:
    st.session_state.ai_analyzing = False

# Create the canvas - this should always be available
canvas_result = st_canvas(
    fill_color=fill_color,
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=background_color,
    width=canvas_width,
    height=canvas_height,
    drawing_mode=drawing_mode,
    key="persistent_canvas",  # Make canvas persistent
    update_streamlit=True,
    display_toolbar=True,
)

# Store canvas data in session state when available
try:
    if hasattr(canvas_result, "image_data") and canvas_result.image_data is not None:
        # Only update canvas_data if no uploaded image is being processed
        if not st.session_state.get("uploaded_image_processed", False):
            st.session_state.canvas_data = canvas_result.image_data
            st.session_state.canvas_initialized = True
            # Also create a backup copy
            st.session_state.canvas_backup = canvas_result.image_data.copy()
except NameError:
    # canvas_result not defined yet, skip this section
    pass

# Simple Canvas Display - Show uploaded image or canvas drawing
st.subheader("📱 Canvas Display:")

# Create a container for the canvas with analysis animation overlay
canvas_container = st.container()

with canvas_container:
    # Show uploaded image if available, otherwise show current canvas
    if uploaded_file is not None:
        # Display uploaded image
        uploaded_file.seek(0)
        uploaded_image = Image.open(uploaded_file)
        if uploaded_image.mode != "RGB":
            uploaded_image = uploaded_image.convert("RGB")

        # Resize to fit canvas dimensions
        uploaded_image = uploaded_image.resize(
            (canvas_width, canvas_height), Image.LANCZOS
        )

        st.image(
            uploaded_image,
            caption="Your Uploaded Sketch/Image",
            width=800,
            use_container_width=False,
        )
        st.info("🎨 This image is now your canvas! You can use AI analysis on it.")

        # Clear button for uploaded image
        if st.button("🗑️ Clear Image", key="clear_image"):
            st.rerun()

    elif hasattr(canvas_result, "image_data") and canvas_result.image_data is not None:
        # Show current canvas drawing
        canvas_array = canvas_result.image_data.astype("uint8")
        display_image = Image.fromarray(canvas_array)
        st.image(
            display_image,
            caption="Live Drawing Preview",
            width=800,
            use_container_width=False,
        )
    else:
        # Show empty canvas message
        st.info("🎨 Upload an image or start drawing to see your canvas here!")

    # Add analysis animation overlay when AI is analyzing
    if st.session_state.get("ai_analyzing", False):
        st.info("🔍 AI Analysis in progress...")
        show_analysis_animation(canvas_container, "AI Analyzing Sketch...")

# Simple image handling - no complex session state management
input_image = None
if uploaded_file is not None:
    # Handle uploaded image
    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Maintain aspect ratio by resizing to fit within canvas dimensions
        img.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
        input_image = img

        # Store for AI analysis
        st.session_state.uploaded_image_bytes = image_bytes

    except Exception as e:
        st.error(f"❌ Error loading uploaded image: {str(e)}")
elif hasattr(canvas_result, "image_data") and canvas_result.image_data is not None:
    # Handle canvas drawing
    canvas_array = canvas_result.image_data.astype("uint8")
    input_image = Image.fromarray(canvas_array)

    # Ensure the image has the correct dimensions
    if input_image.size != (canvas_width, canvas_height):
        input_image = input_image.resize((canvas_width, canvas_height), Image.LANCZOS)

    # Convert to RGB mode to ensure compatibility
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

# Note: Canvas clearing is now handled in the unified canvas display above

CANVAS_SIZE = 800  # Match your st_canvas width


if input_image is not None:
    # Store input image in session state for AI analysis
    st.session_state.input_image = input_image

    # Use custom CSS to ensure no extra padding/margin
    st.markdown(
        """
    <style>
    .stImage > img {
        margin: 0 !important;
        padding: 0 !important;
        max-width: none !important;
        width: 800px !important;
        height: 600px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Hide separate input preview to keep canvas as the main workspace

    # Sketch Type Selection with Buttons (using session state)
    st.subheader("📝 Sketch Type Selection")
    st.markdown("Select the type of sketch you're working with:")

    # Initialize session state for sketch type
    if "sketch_type" not in st.session_state:
        st.session_state.sketch_type = "face"

    # Create columns for buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "👤 Face",
            key="face_button",
            help="Optimized for face portraits and headshots",
            type="primary" if st.session_state.sketch_type == "face" else "secondary",
        ):
            st.session_state.sketch_type = "face"

    with col2:
        if st.button(
            "🏃 Full Body",
            key="body_button",
            help="Optimized for full body figures and poses",
            type=(
                "primary"
                if st.session_state.sketch_type == "full_body"
                else "secondary"
            ),
        ):
            st.session_state.sketch_type = "full_body"

    # Show current selection
    if st.session_state.sketch_type == "face":
        st.success("✅ Face mode selected - Using face-optimized parameters")
    else:
        st.success("✅ Full body mode selected - Using body-optimized parameters")

    # Use session state for sketch type
    sketch_type = st.session_state.sketch_type

    # Style Selection
    st.subheader("🎨 Style Selection")
    st.markdown("Choose the artistic style for your generation:")

    # Initialize session state for style
    if "art_style" not in st.session_state:
        st.session_state.art_style = "realistic"

    # Create columns for style buttons
    col3, col4, col5 = st.columns(3)

    with col3:
        if st.button(
            "🎭 Realistic",
            key="realistic_button",
            help="Photorealistic, natural lighting, detailed features",
            type=(
                "primary" if st.session_state.art_style == "realistic" else "secondary"
            ),
        ):
            st.session_state.art_style = "realistic"

    with col4:
        if st.button(
            "🎨 Cartoon/Animation",
            key="cartoon_button",
            help="Anime, cartoon, stylized, vibrant colors",
            type="primary" if st.session_state.art_style == "cartoon" else "secondary",
        ):
            st.session_state.art_style = "cartoon"

    with col5:
        if st.button(
            "📸 Ultra Realistic",
            key="ultra_realistic_button",
            help="Photorealistic human with sketch detail preservation (hair color, facial features, poses)",
            type=(
                "primary"
                if st.session_state.art_style == "ultra_realistic"
                else "secondary"
            ),
        ):
            st.session_state.art_style = "ultra_realistic"

    # Show current style selection
    if st.session_state.art_style == "realistic":
        st.success("✅ Realistic style selected - Photorealistic generation")
    elif st.session_state.art_style == "cartoon":
        st.success("✅ Cartoon/Animation style selected - Stylized generation")
    else:
        st.success(
            "✅ Ultra Realistic style selected - Photorealistic human with sketch detail preservation"
        )

    # Use session state for art style
    art_style = st.session_state.art_style

    # Advanced Details Tab - Hide complex parameter information for cleaner interface
    with st.expander("🔧 Advanced Details", expanded=False):
        st.subheader("📊 Parameter Preview")
        st.markdown("**Current settings will use these parameters:**")
        st.markdown(
            "*This shows exactly what parameters will be applied based on your selections above.*"
        )

        # Get current preset choice
        current_preset = st.session_state.get("preset_choice", "Default (Recommended)")

        # Show parameters based on current selections
        # Use centralized parameter system for automatic sync
        try:
            params_dict, control_type = get_parameters_from_preset(
                art_style, sketch_type, current_preset
            )

            # Display the parameters dynamically
            if art_style == "realistic":
                if sketch_type == "face":
                    if "Creative" in current_preset:
                        st.info(
                            f"🎯 **Face Realistic Creative**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                        )
                    elif "Balanced" in current_preset:
                        st.info(
                            f"🎯 **Face Realistic Balanced**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                        )
                    else:
                        st.info(
                            f"🎯 **Face Realistic Default**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                        )
                else:  # full_body
                    if "Creative" in current_preset:
                        st.info(
                            f"🎯 **Full Body Realistic Creative**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                        )
                    elif "Balanced" in current_preset:
                        st.info(
                            f"🎯 **Full Body Realistic Balanced**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                        )
                    else:
                        st.info(
                            f"🎯 **Full Body Realistic Default**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                        )
            elif art_style == "cartoon":
                if "Creative" in current_preset:
                    st.info(
                        f"🎯 **Cartoon Creative**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                    )
                elif "Balanced" in current_preset:
                    st.info(
                        f"🎯 **Cartoon Balanced**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                    )
                else:
                    st.info(
                        f"🎯 **Cartoon Default**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Scribble ControlNet"
                    )
            else:  # ultra_realistic
                if "Creative" in current_preset:
                    st.info(
                        f"🎯 **Ultra Realistic Creative**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Canny ControlNet"
                    )
                elif "Balanced" in current_preset:
                    st.info(
                        f"🎯 **Ultra Realistic Balanced**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Canny ControlNet"
                    )
                else:
                    st.info(
                        f"🎯 **Ultra Realistic Default**: ControlNet={params_dict['controlnet_conditioning_scale']}, Guidance={params_dict['guidance_scale']}, Steps={params_dict['num_inference_steps']}, Canny ControlNet"
                    )
        except Exception as e:
            st.error(f"⚠️ Error getting parameters: {e}")
            st.info("🎯 Using fallback parameters")

        # Add a quick reference table
        with st.expander(
            "📋 All Parameter Combinations (Quick Reference)", expanded=False
        ):
            st.markdown("**Realistic Mode:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Face:**")
                st.markdown(
                    f"- Default: CN={PRESET_PARAMETERS['realistic']['face']['Default']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['realistic']['face']['Default']['guidance_scale']}, S={PRESET_PARAMETERS['realistic']['face']['Default']['num_inference_steps']}"
                )
                st.markdown(
                    f"- Balanced: CN={PRESET_PARAMETERS['realistic']['face']['Balanced']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['realistic']['face']['Balanced']['guidance_scale']}, S={PRESET_PARAMETERS['realistic']['face']['Balanced']['num_inference_steps']}"
                )
                st.markdown(
                    f"- Creative: CN={PRESET_PARAMETERS['realistic']['face']['Creative']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['realistic']['face']['Creative']['guidance_scale']}, S={PRESET_PARAMETERS['realistic']['face']['Creative']['num_inference_steps']}"
                )
            with col2:
                st.markdown("**Full Body:**")
                st.markdown(
                    f"- Default: CN={PRESET_PARAMETERS['realistic']['full_body']['Default']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['realistic']['full_body']['Default']['guidance_scale']}, S={PRESET_PARAMETERS['realistic']['full_body']['Default']['num_inference_steps']}"
                )
                st.markdown(
                    f"- Balanced: CN={PRESET_PARAMETERS['realistic']['full_body']['Balanced']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['realistic']['full_body']['Balanced']['guidance_scale']}, S={PRESET_PARAMETERS['realistic']['full_body']['Balanced']['num_inference_steps']}"
                )
                st.markdown(
                    f"- Creative: CN={PRESET_PARAMETERS['realistic']['full_body']['Creative']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['realistic']['full_body']['Creative']['guidance_scale']}, S={PRESET_PARAMETERS['realistic']['full_body']['Creative']['num_inference_steps']}"
                )

            st.markdown("**Cartoon Mode:**")
            st.markdown(
                f"- Default: CN={PRESET_PARAMETERS['cartoon']['face']['Default']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['cartoon']['face']['Default']['guidance_scale']}, S={PRESET_PARAMETERS['cartoon']['face']['Default']['num_inference_steps']}"
            )
            st.markdown(
                f"- Balanced: CN={PRESET_PARAMETERS['cartoon']['face']['Balanced']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['cartoon']['face']['Balanced']['guidance_scale']}, S={PRESET_PARAMETERS['cartoon']['face']['Balanced']['num_inference_steps']}"
            )
            st.markdown(
                f"- Creative: CN={PRESET_PARAMETERS['cartoon']['face']['Creative']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['cartoon']['face']['Creative']['guidance_scale']}, S={PRESET_PARAMETERS['cartoon']['face']['Creative']['num_inference_steps']}"
            )

            st.markdown("**Ultra Realistic Mode:**")
            st.markdown(
                f"- Default: CN={PRESET_PARAMETERS['ultra_realistic']['face']['Default']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['ultra_realistic']['face']['Default']['guidance_scale']}, S={PRESET_PARAMETERS['ultra_realistic']['face']['Default']['num_inference_steps']}"
            )
            st.markdown(
                f"- Balanced: CN={PRESET_PARAMETERS['ultra_realistic']['face']['Balanced']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['ultra_realistic']['face']['Balanced']['guidance_scale']}, S={PRESET_PARAMETERS['ultra_realistic']['face']['Balanced']['num_inference_steps']}"
            )
            st.markdown(
                f"- Creative: CN={PRESET_PARAMETERS['ultra_realistic']['face']['Creative']['controlnet_conditioning_scale']}, G={PRESET_PARAMETERS['ultra_realistic']['face']['Creative']['guidance_scale']}, S={PRESET_PARAMETERS['ultra_realistic']['face']['Creative']['num_inference_steps']}"
            )

            st.markdown("*CN=ControlNet Scale, G=Guidance Scale, S=Steps*")

        st.markdown("---")

        # ControlNet Type Information
        st.markdown("**🔧 ControlNet Types Used:**")
        st.markdown(
            "- **Scribble**: For realistic/cartoon modes - follows your sketch lines"
        )
        st.markdown("- **Canny**: For ultra-realistic mode - preserves edge details")
        st.markdown(
            "- **Scribble**: For realistic creative mode - enhanced creativity with sketch adherence"
        )

    # Preset selection UI per mode (integrated with face/full-body and style)
    st.subheader("🧪 Parameter Presets")
    if art_style == "ultra_realistic":
        preset_help = (
            "Detailed works best for scans/paintings. Balanced for simple clean sketches. "
            "Creative allows larger deviation for very simple stick/line drawings."
        )
        preset_options = [
            "Default (Detailed - Current)",
            "Balanced (Simple Sketch)",
            "Creative (Very Simple)",
        ]
        default_preset_for_style = "Default (Detailed - Current)"
    elif art_style == "realistic":
        preset_help = (
            "Default is your previously best-performing setup. Balanced softens control for smoother realism. "
            "Creative loosens control for more natural variation."
        )
        preset_options = ["Default (Recommended)", "Balanced", "Creative"]
        default_preset_for_style = "Default (Recommended)"
    else:
        preset_help = (
            "Default tightly follows the sketch with clean, crisp colors (great for Pixar/toy-like looks). "
            "Balanced softens the guidance a bit. Creative adds freedom and vibrant shading."
        )
        preset_options = ["Default (Recommended)", "Balanced", "Creative"]
        default_preset_for_style = "Default (Recommended)"

    # Initialize preset once per style change BEFORE creating the widget
    if "_preset_style" not in st.session_state:
        st.session_state._preset_style = art_style
    if (
        "preset_choice" not in st.session_state
        or st.session_state._preset_style != art_style
    ):
        st.session_state.preset_choice = default_preset_for_style
        st.session_state._preset_style = art_style

    # Create the selectbox bound to session state (no assignment to session_state here)
    col1, col2 = st.columns([3, 1])
    with col1:
        preset_choice = st.selectbox(
            "Choose a preset",
            options=preset_options,
            help=preset_help,
            key="preset_choice",
        )
    with col2:
        if st.button("🔄 Reset", help="Reset to default parameters"):
            st.session_state.preset_choice = default_preset_for_style
            st.rerun()

    # Friendly descriptions to avoid confusion
    with st.expander("What do these parameters do?", expanded=False):
        st.markdown(
            "- **ControlNet Scale**: Higher = follows your sketch more; lower = more creative freedom.\n"
            "- **Guidance Scale**: Higher = stronger prompt adherence (can look flatter); lower = more natural texture.\n"
            "- **Steps**: More steps = more detail (slower).\n"
            "- **Control Guidance End** (ultra realistic): Lower ends ControlNet earlier → more realism; higher preserves sketch longer.\n"
            "- **Refiner** (ultra realistic): Adds fine texture and realism after the main image is generated."
        )
        st.markdown(
            "- **Use-cases**:\n"
            "  - Pixar/toy-like 3D look → Cartoon: Default or Balanced.\n"
            "  - CG-looking realistic human → Realistic: Default.\n"
            "  - Ultra-real from detailed scans → Ultra Realistic: Default (Detailed - Current).\n"
            "  - Ultra-real from very simple lines → Ultra Realistic: Creative."
        )

    # Recommendations block with Apply action
    with st.expander("Show recommendation", expanded=False):
        if art_style == "ultra_realistic":
            rec = (
                "Default (Detailed - Current)"
                if sketch_type == "face"
                else "Default (Detailed - Current)"
            )
            rec_reason = "Detailed scans/paintings. For simple line art, try Balanced; for very simple stick lines, Creative."
        elif art_style == "realistic":
            rec = "Default (Recommended)"
            rec_reason = (
                "Best general CG-real look while respecting your sketch pose and hair."
            )
        else:
            rec = "Default (Recommended)"
            rec_reason = "Pixar/toy-like output that traces lines crisply; Balanced softens, Creative adds flair."
        st.markdown(f"**Suggested preset:** {rec}\n\n{rec_reason}")
        if st.button("Apply suggestion"):
            if st.session_state.preset_choice != rec:
                st.session_state.preset_choice = rec
                st.rerun()

    # Check if auto-generation should be triggered after Smart AI analysis
    auto_generate = st.session_state.get("trigger_auto_generation", False)
    if auto_generate:
        # Clear the flag to prevent repeated auto-generation
        st.session_state.trigger_auto_generation = False
        st.info("🤖 Smart AI is auto-generating your painting...")

    if st.button("Generate Painting", type="primary") or auto_generate:
        # Show realistic SVG paintbrush animation using components.html
        import streamlit.components.v1 as components

        # Prepare paintbrush PNG as base64 for HTML embedding
        brush_path = os.path.join(os.path.dirname(__file__), "paintbrush.png")
        with open(brush_path, "rb") as img_file:
            brush_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        brush_img_tag = (
            f'<img src="data:image/png;base64,{brush_b64}" class="paintbrush-img" />'
        )

        paintbrush_animation_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            text-align: center;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .loading-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            padding: 50px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2), 0 0 0 1px rgba(255,255,255,0.1);
            position: relative;
            overflow: hidden;
            max-width: 800px;
            margin: 0 auto;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .loading-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }}
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        .loading-text {{
            font-size: 28px;
            color: #FF6B6B;
            margin-bottom: 40px;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
            z-index: 10;
            animation: colorBounce 2s infinite alternate, colorShift 4s infinite linear;
            display: inline-block;
        }}
        @keyframes colorBounce {{
            0%, 100% {{ transform: translateY(0) scale(1); }}
            20% {{ transform: translateY(-8px) scale(1.05); }}
            40% {{ transform: translateY(0) scale(1); }}
            60% {{ transform: translateY(-4px) scale(1.03); }}
            80% {{ transform: translateY(0) scale(1); }}
        }}
        @keyframes colorShift {{
            0% {{ color: #FF6B6B; }}
            25% {{ color: #FFD700; }}
            50% {{ color: #4ECDC4; }}
            75% {{ color: #45B7D1; }}
            100% {{ color: #FF6B6B; }}
        }}
        .painting-studio {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 80px;
            margin: 40px 0;
            position: relative;
            z-index: 5;
        }}
        .paintbrush {{
            animation: paintbrushMove 12s infinite ease-in-out;
            transform-origin: center bottom;
            filter: drop-shadow(0 5px 15px rgba(0,0,0,0.3));
            position: relative;
            z-index: 20;
        }}
        @keyframes paintbrushMove {{
            0% {{ transform: translateX(-10px) translateY(-5px) rotate(-10deg) scale(1); }}
            8% {{ transform: translateX(30px) translateY(25px) rotate(5deg) scale(1.05); }}
            16% {{ transform: translateX(40px) translateY(35px) rotate(10deg) scale(1.1); }}
            24% {{ transform: translateX(35px) translateY(40px) rotate(5deg) scale(1.05); }}
            32% {{ transform: translateX(45px) translateY(30px) rotate(15deg) scale(1.1); }}
            40% {{ transform: translateX(25px) translateY(35px) rotate(-5deg) scale(1.05); }}
            48% {{ transform: translateX(20px) translateY(30px) rotate(-10deg) scale(1.1); }}
            56% {{ transform: translateX(-20px) translateY(20px) rotate(-15deg) scale(1.05); }}
            64% {{ transform: translateX(-30px) translateY(15px) rotate(-20deg) scale(1.1); }}
            72% {{ transform: translateX(30px) translateY(25px) rotate(15deg) scale(1.05); }}
            80% {{ transform: translateX(40px) translateY(20px) rotate(20deg) scale(1.1); }}
            88% {{ transform: translateX(-25px) translateY(35px) rotate(-10deg) scale(1.05); }}
            96% {{ transform: translateX(-35px) translateY(30px) rotate(-25deg) scale(1.1); }}
            100% {{ transform: translateX(-10px) translateY(-5px) rotate(-10deg) scale(1); }}
        }}
        .color-palette {{
            animation: paletteRotate 10s infinite ease-in-out;
            filter: drop-shadow(0 8px 20px rgba(0,0,0,0.2));
            position: relative;
            z-index: 10;
        }}
        @keyframes paletteRotate {{
            0% {{ transform: rotate(0deg) scale(0.95); }}
            50% {{ transform: rotate(180deg) scale(1.05); }}
            100% {{ transform: rotate(360deg) scale(0.95); }}
        }}
        /* Paint trail from brush */
        .paint-trail {{
            position: absolute;
            width: 3px;
            background: linear-gradient(to bottom, var(--trail-color), transparent);
            border-radius: 2px;
            animation: paintTrail 4s infinite ease-in-out;
            z-index: 7;
        }}
        .trail-1 {{
            --trail-color: #FF6B6B;
            height: 25px;
            left: 45px;
            top: 60px;
            animation-delay: 0.5s;
        }}
        .trail-2 {{
            --trail-color: #4ECDC4;
            height: 20px;
            left: 48px;
            top: 65px;
            animation-delay: 1.5s;
        }}
        .trail-3 {{
            --trail-color: #45B7D1;
            height: 30px;
            left: 42px;
            top: 55px;
            animation-delay: 2.5s;
        }}
        @keyframes paintTrail {{
            0% {{ height: 0px; opacity: 0; }}
            20% {{ height: var(--trail-height, 25px); opacity: 1; }}
            80% {{ height: var(--trail-height, 25px); opacity: 1; }}
            100% {{ height: 0px; opacity: 0; }}
        }}
        /* Dynamic paint splashes that change colors and sizes */
        .paint-splash {{
            position: absolute;
            border-radius: 50%;
            animation: splashFloat 8s infinite ease-in-out;
            filter: blur(2px) drop-shadow(0 4px 8px rgba(0,0,0,0.2));
            z-index: 1;
            opacity: 0.7;
        }}
        .splash-1 {{ 
            background: #FF6B6B; /* Red */
            width: 80px; height: 80px; 
            top: 15%; left: 15%; 
            animation-delay: 0s; 
            animation: splashFloat 8s infinite ease-in-out, colorChange1 6s infinite ease-in-out, sizeChange1 4s infinite ease-in-out;
        }}
        .splash-2 {{ 
            background: #4ECDC4; /* Green */
            width: 60px; height: 60px; 
            top: 20%; right: 20%; 
            animation-delay: 2s; 
            animation: splashFloat 8s infinite ease-in-out, colorChange2 6s infinite ease-in-out, sizeChange2 4s infinite ease-in-out;
        }}
        .splash-3 {{ 
            background: #FFE66D; /* Yellow */
            width: 70px; height: 70px; 
            bottom: 20%; left: 20%; 
            animation-delay: 4s; 
            animation: splashFloat 8s infinite ease-in-out, colorChange3 6s infinite ease-in-out, sizeChange3 4s infinite ease-in-out;
        }}
        .splash-4 {{ 
            background: #45B7D1; /* Blue */
            width: 90px; height: 90px; 
            bottom: 15%; right: 25%; 
            animation-delay: 1s; 
            animation: splashFloat 8s infinite ease-in-out, colorChange4 6s infinite ease-in-out, sizeChange4 4s infinite ease-in-out;
        }}
        /* Color changing animations */
        @keyframes colorChange1 {{
            0%, 100% {{ background: #FF6B6B; }} /* Red */
            25% {{ background: #FF8E8E; }} /* Light red */
            50% {{ background: #FFB6C1; }} /* Pink */
            75% {{ background: #FF6B6B; }} /* Back to red */
        }}
        @keyframes colorChange2 {{
            0%, 100% {{ background: #4ECDC4; }} /* Green */
            25% {{ background: #7FFFD4; }} /* Light green */
            50% {{ background: #98FB98; }} /* Pale green */
            75% {{ background: #4ECDC4; }} /* Back to green */
        }}
        @keyframes colorChange3 {{
            0%, 100% {{ background: #FFE66D; }} /* Yellow */
            25% {{ background: #FFFFE0; }} /* Light yellow */
            50% {{ background: #F0E68C; }} /* Khaki */
            75% {{ background: #FFE66D; }} /* Back to yellow */
        }}
        @keyframes colorChange4 {{
            0%, 100% {{ background: #45B7D1; }} /* Blue */
            25% {{ background: #87CEEB; }} /* Light blue */
            50% {{ background: #B0E0E6; }} /* Powder blue */
            75% {{ background: #45B7D1; }} /* Back to blue */
        }}
        /* Size changing animations */
        @keyframes sizeChange1 {{
            0%, 100% {{ width: 80px; height: 80px; }}
            50% {{ width: 120px; height: 120px; }}
        }}
        @keyframes sizeChange2 {{
            0%, 100% {{ width: 60px; height: 60px; }}
            50% {{ width: 100px; height: 100px; }}
        }}
        @keyframes sizeChange3 {{
            0%, 100% {{ width: 70px; height: 70px; }}
            50% {{ width: 110px; height: 110px; }}
        }}
        @keyframes sizeChange4 {{
            0%, 100% {{ width: 90px; height: 90px; }}
            50% {{ width: 130px; height: 130px; }}
        }}
        @keyframes splashFloat {{
            0%, 100% {{ 
                transform: translateY(0px) scale(1) rotate(var(--rotation, 0deg)); 
                opacity: 0.5; 
            }}
            50% {{ 
                transform: translateY(-25px) scale(1.3) rotate(var(--rotation, 0deg)); 
                opacity: 0.9; 
            }}
        }}
        .progress-indicator {{
            margin-top: 30px;
            position: relative;
            z-index: 10;
        }}
        .progress-bar {{
            width: 100%;
            height: 12px;
            background: rgba(255,107,107,0.2);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
            border: 2px solid rgba(255,255,255,0.3);
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #FFE66D);
            border-radius: 4px;
            animation: progressMove 3s infinite ease-in-out;
            background-size: 200% 100%;
            box-shadow: 0 0 10px rgba(255,107,107,0.5);
        }}
        @keyframes progressMove {{
            0% {{ width: 0%; background-position: 0% 50%; }}
            50% {{ width: 70%; background-position: 100% 50%; }}
            100% {{ width: 100%; background-position: 0% 50%; }}
        }}
        .status-dots {{
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-top: 20px;
        }}
        .status-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            animation: dotPulse 1.5s infinite ease-in-out;
            box-shadow: 0 0 10px rgba(255,107,107,0.5);
        }}
        .status-dot:nth-child(2) {{ animation-delay: 0.3s; background: linear-gradient(45deg, #4ECDC4, #45B7D1); }}
        .status-dot:nth-child(3) {{ animation-delay: 0.6s; background: linear-gradient(45deg, #45B7D1, #FFE66D); }}
        @keyframes dotPulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.6; }}
            50% {{ transform: scale(1.8); opacity: 1; }}
        }}
        /* Professional decorative elements */
        .decorative-elements {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 10;
        }}
        .decorative-star {{
            display: inline-block;
            width: 20px;
            height: 20px;
            background: #FFD700;
            clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
            animation: starTwinkle 2s infinite ease-in-out;
            margin: 0 5px;
        }}
        .decorative-star:nth-child(2) {{ animation-delay: 0.5s; }}
        .decorative-star:nth-child(3) {{ animation-delay: 1s; }}
        @keyframes starTwinkle {{
            0%, 100% {{ transform: scale(1) rotate(0deg); opacity: 0.7; }}
            50% {{ transform: scale(1.3) rotate(180deg); opacity: 1; }}
        }}
        .paintbrush-img {{
            width: 100px;
            height: auto;
            filter: drop-shadow(0 5px 15px rgba(0,0,0,0.3));
            animation: paintbrushMove 10s infinite alternate;
            transform-origin: center bottom;
            z-index: 21;
        }}
        @keyframes paintbrushMove {{
            0%   {{ transform: translate(0px, 0px) rotate(-20deg) scale(1); }}
            10%  {{ transform: translate(-30px, 20px) rotate(-10deg) scale(1.03); }}
            20%  {{ transform: translate(40px, -15px) rotate(8deg) scale(1.07); }}
            35%  {{ transform: translate(-40px, 30px) rotate(-12deg) scale(1.05); }}
            45%  {{ transform: translate(60px, -10px) rotate(0deg) scale(1.12); }} /* hover over color plate */
            55%  {{ transform: translate(60px, -10px) rotate(0deg) scale(1.12); }} /* pause over color plate */
            65%  {{ transform: translate(50px, 20px) rotate(12deg) scale(1.09); }}
            80%  {{ transform: translate(30px, 35px) rotate(8deg) scale(1.07); }}
            100% {{ transform: translate(0px, 0px) rotate(-20deg) scale(1); }}
        }}
        </style>
        </head>
        <body>
        <div class="loading-container">
            <div class="decorative-elements">
                <div class="decorative-star"></div>
                <div class="decorative-star"></div>
                <div class="decorative-star"></div>
            </div>
            <div class="loading-text">🎨 Creating your masterpiece...</div>
            <div class="painting-studio">
                <!-- Paintbrush (left, in front if overlap) -->
                <div class="paintbrush">
                    {brush_img_tag}
                </div>
                <!-- Color Palette (right) -->
                <div class="color-palette">
                    <svg width="140" height="140" viewBox="0 0 140 140">
                        <!-- Palette base with wood texture -->
                        <defs>
                            <linearGradient id="paletteGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#A0522D;stop-opacity:1" />
                                <stop offset="50%" style="stop-color:#8B4513;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#654321;stop-opacity:1" />
                            </linearGradient>
                        </defs>
                        
                        <!-- Palette base -->
                        <ellipse cx="70" cy="70" rx="60" ry="40" fill="url(#paletteGradient)" stroke="#654321" stroke-width="4"/>
                        <ellipse cx="70" cy="70" rx="55" ry="35" fill="#A0522D"/>
                        
                        <!-- Paint wells with 3D effect -->
                        <circle cx="50" cy="50" r="12" fill="#FFE66D" stroke="#4682B4" stroke-width="2"/>
                        <circle cx="90" cy="50" r="12" fill="#87CEEB" stroke="#4682B4" stroke-width="2"/>
                        <circle cx="50" cy="90" r="12" fill="#FFB6C1" stroke="#8B0000" stroke-width="2"/>
                        <circle cx="90" cy="90" r="12" fill="#FF8E8E" stroke="#8B0000" stroke-width="2"/>
                        <circle cx="70" cy="70" r="12" fill="#4ECDC4" stroke="#006400" stroke-width="2"/>
                        
                        <!-- Paint texture in wells -->
                        <circle cx="50" cy="50" r="6" fill="#FFFFE0" opacity="0.8"/>
                        <circle cx="90" cy="50" r="6" fill="#B0E0E6" opacity="0.8"/>
                        <circle cx="50" cy="90" r="6" fill="#FFC0CB" opacity="0.8"/>
                        <circle cx="90" cy="90" r="6" fill="#FFB6C1" opacity="0.8"/>
                        <circle cx="70" cy="70" r="6" fill="#7FFFD4" opacity="0.8"/>
                        
                        <!-- Thumb hole -->
                        <ellipse cx="70" cy="30" rx="8" ry="12" fill="#654321"/>
                    </svg>
                </div>
                
                <!-- Dynamic paint splashes that change colors and sizes -->
                <div class="paint-splash splash-1"></div>
                <div class="paint-splash splash-2"></div>
                <div class="paint-splash splash-3"></div>
                <div class="paint-splash splash-4"></div>
            </div>
            
            <!-- Progress indicator -->
            <div class="progress-indicator">
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <div class="status-dots">
                    <div class="status-dot"></div>
                    <div class="status-dot"></div>
                    <div class="status-dot"></div>
                </div>
            </div>
        </div>
        </body>
        </html>
        """

        components.html(paintbrush_animation_html, height=300)

        # Generate the image
        try:
            # Ensure image is in RGB mode
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")

            # Convert to bytes with proper error handling
            img_bytes = image_to_bytes(input_image)

            # Verify the bytes are valid
            if not img_bytes or len(img_bytes) == 0:
                st.error("Failed to convert image to bytes. Please try drawing again.")
                st.stop()

            files = {"image": ("input.png", img_bytes, "image/png")}
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try drawing again or upload a different image.")
            st.stop()
        # Define default negative prompt for all models
        base_negative = "deformed, distorted, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, floating limbs, mutated hands and fingers, out of focus, long neck, long body, mutated hands and fingers, missing arms, missing legs, extra arms, extra legs, malformed limbs, mutated limbs, double image, blurred, ugly, disgusting, amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed, mutated, anatomical, anatomical error, goofy, unrealistic, cartoon, anime, painting, drawing, sketch, illustration, low quality, blurry, pixelated"

        # Smart negative prompt based on context
        context_negative = ""

        # If user mentions boy/child in prompt, prevent adult/woman generation
        if prompt and any(
            word in prompt.lower() for word in ["boy", "child", "kid", "young"]
        ):
            context_negative = "adult, woman, female, girl, old person, elderly, mature"
        # If user mentions girl/woman in prompt, prevent man/boy generation
        elif prompt and any(
            word in prompt.lower() for word in ["girl", "woman", "female", "lady"]
        ):
            context_negative = "man, boy, male, child, kid, young"
        # If user mentions man/male in prompt, prevent woman/girl generation
        elif prompt and any(
            word in prompt.lower() for word in ["man", "male", "guy", "dude"]
        ):
            context_negative = "woman, girl, female, lady"
        # If it's an animal theme, prevent human generation
        elif "animal" in theme.lower() or any(
            word in theme.lower() for word in ["pet", "wildlife", "creature"]
        ):
            context_negative = "human, person, man, woman, child, boy, girl"
        # If it's architecture/object theme, prevent human generation
        elif (
            "architecture" in theme.lower()
            or "object" in theme.lower()
            or any(
                word in theme.lower()
                for word in ["building", "furniture", "vehicle", "product"]
            )
        ):
            context_negative = "human, person, man, woman, child, boy, girl"

        # Combine base and context negative prompts
        negative_prompt = base_negative
        if context_negative:
            negative_prompt += ", " + context_negative

        # Format data based on model
        if model == "ControlNet":
            # Enhanced prompts based on theme - STRICTER for sketch following
            if "portrait" in theme.lower() or "headshot" in theme.lower():
                enhanced_prompt = f"young boy portrait, {prompt}, male child, boy face, child portrait, high quality, detailed face, natural lighting, sharp focus, perfect anatomy, symmetrical face, natural proportions"
            elif "realistic" in theme.lower():
                enhanced_prompt = f"young boy, {prompt}, male child, boy face, realistic human portrait, professional photography, detailed features, natural skin texture, high resolution, beautiful, perfect anatomy, symmetrical face, natural proportions"
            else:
                enhanced_prompt = (
                    f"young boy, {prompt}, male child"
                    if prompt
                    else "young boy, male child"
                )

            data = {"prompt": enhanced_prompt, "negative_prompt": negative_prompt}
        else:
            # Local backend expects model, theme, prompt
            data = {"model": model, "theme": theme, "prompt": prompt}

        # Choose backend URL based on model
        if model == "SDXL + ControlNet (Kaggle)":
            # Use current backend with optimal parameters
            st.info("🎨 Using SDXL + ControlNet Backend with optimal parameters...")

            # Get enhanced prompts from theme system
            enhanced_prompt, enhanced_negative, detected_style = (
                get_enhanced_prompts_for_theme(
                    theme, sketch_type, prompt if prompt else ""
                )
            )

            # OVERRIDE: Use manual style selection instead of automatic theme detection
            # This ensures the user's choice takes priority
            manual_style = st.session_state.get("art_style", "realistic")
            if manual_style != detected_style:
                st.info(
                    f"🎨 Overriding theme detection: Using {manual_style.title()} style (you selected) instead of {detected_style.title()} (theme detected)"
                )
                detected_style = manual_style

                # REGENERATE PROMPT to match the selected style
                if detected_style == "realistic":
                    if sketch_type == "face":
                        enhanced_prompt = f"{prompt if prompt else 'portrait'}, realistic photographic portrait, natural skin texture, professional studio lighting, shallow depth of field, sharp focus, single person, detailed eyes, natural hair, smooth facial geometry, no outline overlay"
                        enhanced_negative = "cartoon, anime, illustration, drawing, painting, 3d render, cgi, doll-like, plastic skin, overprocessed, smooth flat shading, line overlay, heavy outlines, deformed, extra limbs, bad anatomy, multiple faces, blurry, low quality"
                    else:  # full_body
                        enhanced_prompt = f"{prompt if prompt else 'full body portrait'}, realistic full body photo, professional photography, studio or clean backdrop, correct human anatomy, natural pose, single person, detailed clothing folds, smooth body geometry, no outline overlay"
                        enhanced_negative = "cartoon, anime, illustration, drawing, painting, 3d render, cgi, doll-like, plastic skin, overprocessed, smooth flat shading, line overlay, heavy outlines, deformed, extra limbs, bad anatomy, multiple people, blurry, low quality"
                elif detected_style == "cartoon":
                    if sketch_type == "face":
                        enhanced_prompt = f"{prompt if prompt else 'portrait'}, anime style portrait, vibrant colors, stylized features, clean lines, detailed illustration, single face"
                        enhanced_negative = "deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, realistic, photograph"
                    else:  # full_body
                        enhanced_prompt = f"{prompt if prompt else 'full body portrait'}, anime style full body portrait, vibrant colors, stylized features, clean lines, detailed illustration, single figure, dynamic pose"
                        enhanced_negative = "deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people, realistic, photograph"
                else:  # ultra_realistic
                    if sketch_type == "face":
                        enhanced_prompt = f"{prompt if prompt else 'portrait'}, photorealistic portrait, DSLR photography, natural skin texture with pores, detailed facial features, professional studio lighting, sharp focus, beautiful, proper anatomy, ultra high resolution, 8k quality, HDR, bokeh, natural expression, realistic hair strands, detailed eyes and eyebrows, natural skin tone, single person, single face, no group, no crowd"
                        enhanced_negative = (
                            "deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered"
                            if not negative_prompt
                            else f"deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered, {negative_prompt}"
                        )
                    else:  # full_body
                        enhanced_prompt = f"{prompt if prompt else 'full body portrait'}, photorealistic full body portrait, DSLR photography, natural skin texture with pores, detailed facial features, wearing casual clothing, decent attire, appropriate dress, proper anatomy, single figure, natural pose, ultra high resolution, 8k quality, HDR, bokeh, natural expression, realistic hair strands, detailed eyes and eyebrows, natural skin tone, single person, no group, no crowd"
                        enhanced_negative = (
                            "deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered"
                            if not negative_prompt
                            else f"deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered, {negative_prompt}"
                        )

            # Use the detected style from theme instead of manual selection
            st.info(f"🎨 Theme Style: {detected_style.title()}")
            st.info(f"🎨 Enhanced Prompt: {enhanced_prompt}")

            # Check prompt length to avoid truncation
            if hasattr(st, "check_prompt_length"):
                check_prompt_length(enhanced_prompt)
            else:
                # Simple token count warning
                token_estimate = (
                    len(enhanced_prompt.split()) + len(enhanced_prompt) // 4
                )
                if token_estimate > 77:
                    st.warning(
                        f"⚠️ Your prompt is approximately {token_estimate} tokens (limit: 77). Some details may be cut off!"
                    )
                    st.info(
                        "💡 Consider shortening your prompt to prioritize key details like hair color, clothing, etc."
                    )

            # Direct backend call - simpler and more reliable
            try:
                # Get backend URL
                backend_url = get_backend_url()
                if "your-ngrok-url-here" in backend_url:
                    st.error("⚠️ Backend URL not configured")
                    st.stop()

                # Convert image to base64
                image_base64 = base64.b64encode(img_bytes).decode("utf-8")

                # Set parameters based on style and preset
                if detected_style == "realistic":
                    # Use centralized parameter system for realistic mode
                    try:
                        params_dict, control_type = get_parameters_from_preset(
                            detected_style,
                            sketch_type,
                            st.session_state.get("preset_choice", "Default"),
                        )
                        params = params_dict
                    except Exception as e:
                        st.warning(
                            f"⚠️ Using fallback parameters for realistic mode: {e}"
                        )
                        # Fallback to your previous working parameters
                        if sketch_type == "face":
                            params = {
                                "controlnet_conditioning_scale": 0.95,
                                "guidance_scale": 6.5,
                                "num_inference_steps": 30,
                            }
                            control_type = "scribble"
                        else:  # full_body
                            params = {
                                "controlnet_conditioning_scale": 0.60,
                                "guidance_scale": 8.0,
                                "num_inference_steps": 40,
                            }
                            control_type = "scribble"
                elif detected_style == "cartoon":
                    # Cartoon mode - use centralized parameter system
                    try:
                        params_dict, control_type = get_parameters_from_preset(
                            detected_style,
                            sketch_type,
                            st.session_state.get("preset_choice", "Default"),
                        )
                        params = params_dict
                    except Exception as e:
                        st.warning(f"⚠️ Using fallback parameters for cartoon mode: {e}")
                        # Fallback to your previous working parameters
                        if sketch_type == "face":
                            params = {
                                "controlnet_conditioning_scale": 0.95,
                                "guidance_scale": 7.0,
                                "num_inference_steps": 32,
                            }
                        else:  # full_body
                            params = {
                                "controlnet_conditioning_scale": 0.95,
                                "guidance_scale": 7.0,
                                "num_inference_steps": 32,
                            }
                        control_type = "scribble"
                else:
                    # Ultra realistic mode - use centralized parameter system
                    try:
                        params_dict, control_type = get_parameters_from_preset(
                            detected_style,
                            sketch_type,
                            st.session_state.get("preset_choice", "Default"),
                        )
                        params = params_dict
                    except Exception as e:
                        st.warning(
                            f"⚠️ Using fallback parameters for ultra-realistic mode: {e}"
                        )
                        # Fallback to your previous working parameters
                        params = {
                            "controlnet_conditioning_scale": 0.95,
                            "guidance_scale": 7.0,
                            "num_inference_steps": 32,
                        }
                        control_type = "canny"

                # Prepare request
                json_data = {
                    "prompt": enhanced_prompt,
                    "negative_prompt": enhanced_negative,
                    "image_data": image_base64,
                    "control_type": control_type,
                    "sketch_type": sketch_type,
                    "art_style": detected_style,
                    "num_inference_steps": params["num_inference_steps"],
                    "controlnet_conditioning_scale": params[
                        "controlnet_conditioning_scale"
                    ],
                    "guidance_scale": params["guidance_scale"],
                    "control_guidance_start": 0.0,
                    "control_guidance_end": 0.8,
                    "use_refiner": False,
                }

                # Make the API call
                start_time = time.time()

                # Add retry logic for connection issues
                max_retries = 3
                try:
                    for attempt in range(max_retries):
                        try:
                            with st.spinner(
                                f"🎨 Generating {sketch_type.replace('_', ' ')} {art_style} painting... (Attempt {attempt + 1}/{max_retries})"
                            ):
                                st.info(f"🔍 Sending request to backend...")
                                st.info(
                                    f"📊 Parameters: ControlNet={params['controlnet_conditioning_scale']}, Guidance={params['guidance_scale']}, Steps={params['num_inference_steps']}"
                                )
                                st.info(f"🎯 Control Type: {control_type}")

                                response = requests.post(
                                    f"{get_base_url()}/generate",
                                    json=json_data,
                                    timeout=600,  # Extended timeout for all SDXL modes
                                )

                            # If we get here, the request was successful
                            st.success(f"✅ Request completed successfully!")
                            break

                        except requests.exceptions.Timeout:
                            if attempt < max_retries - 1:
                                st.warning(
                                    f"⏰ Attempt {attempt + 1} timed out. Retrying..."
                                )
                                st.info(
                                    "💡 This usually means the backend is processing a complex image"
                                )
                                time.sleep(2)  # Wait before retry
                                continue
                            else:
                                st.error(
                                    "⏰ All attempts timed out. The backend is processing a complex image."
                                )
                                st.info(
                                    "💡 Try reducing image complexity or using a simpler preset."
                                )
                                st.info(
                                    "💡 Check your Kaggle backend logs for memory issues"
                                )
                                st.stop()

                        except requests.exceptions.ConnectionError as e:
                            if attempt < max_retries - 1:
                                st.warning(
                                    f"🔌 Attempt {attempt + 1} failed due to connection error: {str(e)}"
                                )
                                st.info(
                                    "💡 This usually means the backend crashed or ran out of memory"
                                )
                                st.info(
                                    "💡 Check your Kaggle backend for error messages"
                                )
                                time.sleep(3)  # Wait longer for connection issues
                                continue
                            else:
                                st.error(
                                    "🔌 Connection error after all retries. Please check if the backend is running."
                                )
                                st.error(f"🔍 Error details: {str(e)}")
                                st.info(
                                    "💡 Ensure your ngrok URL is up to date and backend is accessible."
                                )
                                st.info(
                                    "💡 Check Kaggle backend logs for crashes or memory issues"
                                )
                                st.stop()

                        except Exception as e:
                            if attempt < max_retries - 1:
                                st.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                                st.info("💡 Unexpected error occurred, retrying...")
                                time.sleep(2)
                                continue
                            else:
                                st.error(f"❌ All attempts failed with error: {str(e)}")
                                st.stop()

                except Exception as e:
                    st.error(f"❌ Retry logic failed: {str(e)}")
                    st.stop()

                # Process response after successful request
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        # Convert base64 image back to PIL
                        image_data = base64.b64decode(result["image"])
                        result_image = Image.open(io.BytesIO(image_data))

                        # Show success message
                        if (
                            detected_style == "realistic"
                            and "Creative"
                            in st.session_state.get("preset_choice", "Default")
                        ):
                            st.success(
                                f"✅ {sketch_type.title()} realistic generation successful!"
                            )
                            st.info(
                                f"🎯 Using {control_type.title()} ControlNet for optimal results"
                            )
                        else:
                            st.success(
                                f"✅ {sketch_type.title()} {detected_style} generation successful!"
                            )

                            st.info(
                                f"📊 Used optimal parameters: ControlNet={params['controlnet_conditioning_scale']}, Guidance={params['guidance_scale']}"
                            )

                            # Add download button
                            buf = io.BytesIO()
                            result_image.save(buf, format="PNG")
                            st.download_button(
                                label="💾 Download Image",
                                data=buf.getvalue(),
                                file_name=f"{detected_style}_{sketch_type}_{int(time.time())}.png",
                                mime="image/png",
                            )

                            # Display the result
                            st.image(
                                result_image,
                                caption=f"SDXL + ControlNet {detected_style.title()} Result",
                                use_container_width=True,
                            )

                            # Show curtain reveal
                            show_curtain_reveal(result_image)
                    elif result.get("status") != "success":
                        st.error(
                            f"❌ Backend error: {result.get('detail', 'Unknown error')}"
                        )
                        st.stop()
                else:
                    st.error(f"❌ Backend error: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    st.stop()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

        elif model == "SDXL + ControlNet Official (Kaggle)":
            # Use current backend with optimal parameters
            st.info("🎨 Using SDXL + ControlNet Backend with optimal parameters...")

            # Get enhanced prompts from theme system
            enhanced_prompt, enhanced_negative, detected_style = (
                get_enhanced_prompts_for_theme(
                    theme, sketch_type, prompt if prompt else ""
                )
            )

            # OVERRIDE: Use manual style selection instead of automatic theme detection
            # This ensures the user's choice takes priority
            manual_style = st.session_state.get("art_style", "realistic")
            if manual_style != detected_style:
                st.info(
                    f"🎨 Overriding theme detection: Using {manual_style.title()} style (you selected) instead of {detected_style.title()} (theme detected)"
                )
                detected_style = manual_style

                # REGENERATE PROMPT to match the selected style
                if detected_style == "realistic":
                    if sketch_type == "face":
                        enhanced_prompt = f"{prompt if prompt else 'portrait'}, realistic photographic portrait, natural skin texture, professional studio lighting, shallow depth of field, sharp focus, single person, detailed eyes, natural hair, smooth facial geometry, no outline overlay"
                        enhanced_negative = "cartoon, anime, illustration, drawing, painting, 3d render, cgi, doll-like, plastic skin, overprocessed, smooth flat shading, line overlay, heavy outlines, deformed, extra limbs, bad anatomy, multiple faces, blurry, low quality"
                    else:  # full_body
                        enhanced_prompt = f"{prompt if prompt else 'full body portrait'}, realistic full body photo, professional photography, studio or clean backdrop, correct human anatomy, natural pose, single person, detailed clothing folds, smooth body geometry, no outline overlay"
                        enhanced_negative = "cartoon, anime, illustration, drawing, painting, 3d render, cgi, doll-like, plastic skin, overprocessed, smooth flat shading, line overlay, heavy outlines, deformed, extra limbs, bad anatomy, multiple people, blurry, low quality"
                elif detected_style == "cartoon":
                    if sketch_type == "face":
                        enhanced_prompt = f"anime style portrait, vibrant colors, stylized features, clean lines, detailed illustration, single face, {prompt if prompt else 'portrait'}"
                        enhanced_negative = "deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, realistic, photograph"
                    else:  # full_body
                        enhanced_prompt = f"anime style full body portrait, vibrant colors, stylized features, clean lines, detailed illustration, single figure, dynamic pose, {prompt if prompt else 'full body portrait'}"
                        enhanced_negative = "deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people, realistic, photograph"
                else:  # ultra_realistic
                    if sketch_type == "face":
                        enhanced_prompt = f"photorealistic portrait, DSLR photography, natural skin texture with pores, detailed facial features, professional studio lighting, sharp focus, beautiful, proper anatomy, ultra high resolution, 8k quality, HDR, bokeh, natural expression, realistic hair strands, detailed eyes and eyebrows, natural skin tone, single person, single face, no group, no crowd, {prompt if prompt else 'portrait'}"
                        enhanced_negative = (
                            "deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered"
                            if not negative_prompt
                            else f"deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered, {negative_prompt}"
                        )
                    else:  # full_body
                        enhanced_prompt = f"{prompt if prompt else 'full body portrait'}, photorealistic full body portrait, DSLR photography, natural skin texture with pores, detailed facial features, wearing casual clothing, decent attire, appropriate dress, proper anatomy, single figure, natural pose, ultra high resolution, 8k quality, HDR, bokeh, natural expression, realistic hair strands, detailed eyes and eyebrows, natural skin tone, single person, no group, no crowd, {prompt if prompt else 'full body portrait'}"
                        enhanced_negative = (
                            "deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered"
                            if not negative_prompt
                            else f"deformed, extra limbs, multiple faces, multiple people, mutated hands, fused fingers, extra fingers, missing fingers, distorted hands, bad anatomy, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, inappropriate clothing, revealing, nude, naked, cartoon, anime, illustration, painting, drawing, sketch, unrealistic, artificial, digital art, 3d render, CGI, stylized, animated, cell shaded, flat colors, simple shading, exaggerated features, doll-like, toy-like, plastic, artificial skin, smooth skin, perfect skin, airbrushed, retouched, filtered, {negative_prompt}"
                        )

            # Use the detected style from theme instead of manual selection
            st.info(f"🎨 Theme Style: {detected_style.title()}")
            st.info(f"🎨 Enhanced Prompt: {enhanced_prompt}")

            # Check prompt length to avoid truncation
            if hasattr(st, "check_prompt_length"):
                check_prompt_length(enhanced_prompt)
            else:
                # Simple token count warning
                token_estimate = (
                    len(enhanced_prompt.split()) + len(enhanced_prompt) // 4
                )
                if token_estimate > 77:
                    st.warning(
                        f"⚠️ Your prompt is approximately {token_estimate} tokens (limit: 77). Some details may be cut off!"
                    )
                    st.info(
                        "💡 Consider shortening your prompt to prioritize key details like hair color, clothing, etc."
                    )

            # Direct backend call - simpler and more reliable
            try:
                # Get backend URL
                backend_url = get_backend_url()
                if "your-ngrok-url-here" in backend_url:
                    st.error("⚠️ Backend URL not configured")
                    st.stop()

                # Convert image to base64
                image_base64 = base64.b64encode(img_bytes).decode("utf-8")

                # Set parameters based on style and preset
                if detected_style == "realistic":
                    # Use centralized parameter system for realistic mode
                    try:
                        params_dict, control_type = get_parameters_from_preset(
                            detected_style,
                            sketch_type,
                            st.session_state.get("preset_choice", "Default"),
                        )
                        params = params_dict
                    except Exception as e:
                        st.warning(
                            f"⚠️ Using fallback parameters for realistic mode: {e}"
                        )
                        # Fallback to your previous working parameters
                        if sketch_type == "face":
                            if "Creative" in st.session_state.get(
                                "preset_choice", "Default"
                            ):
                                params = {
                                    "controlnet_conditioning_scale": 0.8,
                                    "guidance_scale": 8.0,
                                    "num_inference_steps": 40,
                                }
                                control_type = (
                                    "scribble"  # Use Scribble for enhanced creativity
                                )
                            else:
                                params = {
                                    "controlnet_conditioning_scale": 0.95,
                                    "guidance_scale": 6.5,
                                    "num_inference_steps": 30,
                                }
                                control_type = "scribble"
                        else:  # full_body
                            if "Creative" in st.session_state.get(
                                "preset_choice", "Default"
                            ):
                                params = {
                                    "controlnet_conditioning_scale": 0.8,
                                    "guidance_scale": 7.0,
                                    "num_inference_steps": 40,
                                }
                                control_type = (
                                    "scribble"  # Use Scribble for enhanced creativity
                                )
                            else:
                                params = {
                                    "controlnet_conditioning_scale": 0.60,
                                    "guidance_scale": 8.0,
                                    "num_inference_steps": 40,
                                }
                                control_type = "scribble"
                elif detected_style == "cartoon":
                    # Cartoon mode - use centralized parameter system
                    try:
                        params_dict, control_type = get_parameters_from_preset(
                            detected_style,
                            sketch_type,
                            st.session_state.get("preset_choice", "Default"),
                        )
                        params = params_dict
                    except Exception as e:
                        st.warning(f"⚠️ Using fallback parameters for cartoon mode: {e}")
                        # Fallback to your previous working parameters
                        if sketch_type == "face":
                            params = {
                                "controlnet_conditioning_scale": 0.95,
                                "guidance_scale": 7.0,
                                "num_inference_steps": 32,
                            }
                        else:  # full_body
                            params = {
                                "controlnet_conditioning_scale": 0.95,
                                "guidance_scale": 7.0,
                                "num_inference_steps": 32,
                            }
                        control_type = "scribble"
                else:
                    # Ultra realistic mode - use centralized parameter system
                    try:
                        params_dict, control_type = get_parameters_from_preset(
                            detected_style,
                            sketch_type,
                            st.session_state.get("preset_choice", "Default"),
                        )
                        params = params_dict
                    except Exception as e:
                        st.warning(
                            f"⚠️ Using fallback parameters for ultra-realistic mode: {e}"
                        )
                        # Fallback to your previous working parameters
                        params = {
                            "controlnet_conditioning_scale": 0.95,
                            "guidance_scale": 7.0,
                            "num_inference_steps": 32,
                        }
                        control_type = "canny"

                # Prepare request
                json_data = {
                    "prompt": enhanced_prompt,
                    "negative_prompt": enhanced_negative,
                    "image_data": image_base64,
                    "control_type": control_type,
                    "sketch_type": sketch_type,
                    "art_style": detected_style,
                    "num_inference_steps": params["num_inference_steps"],
                    "controlnet_conditioning_scale": params[
                        "controlnet_conditioning_scale"
                    ],
                    "guidance_scale": params["guidance_scale"],
                    "control_guidance_start": 0.0,
                    "control_guidance_end": 0.8,
                    "use_refiner": False,
                }

                # Make the API call
                start_time = time.time()

                # Add retry logic for connection issues
                max_retries = 3
                try:
                    for attempt in range(max_retries):
                        try:
                            with st.spinner(
                                f"🎨 Generating {sketch_type.replace('_', ' ')} {art_style} painting... (Attempt {attempt + 1}/{max_retries})"
                            ):
                                st.info(f"🔍 Sending request to backend...")
                                st.info(
                                    f"📊 Parameters: ControlNet={params['controlnet_conditioning_scale']}, Guidance={params['guidance_scale']}, Steps={params['num_inference_steps']}"
                                )
                                st.info(f"🎯 Control Type: {control_type}")

                                response = requests.post(
                                    f"{get_base_url()}/generate",
                                    json=json_data,
                                    timeout=600,  # Extended timeout for all SDXL modes
                                )

                            # If we get here, the request was successful
                            st.success(f"✅ Request completed successfully!")
                            break

                        except requests.exceptions.Timeout:
                            if attempt < max_retries - 1:
                                st.warning(
                                    f"⏰ Attempt {attempt + 1} timed out. Retrying..."
                                )
                                st.info(
                                    "💡 This usually means the backend is processing a complex image"
                                )
                                time.sleep(2)  # Wait before retry
                                continue
                            else:
                                st.error(
                                    "⏰ All attempts timed out. The backend is processing a complex image."
                                )
                                st.info(
                                    "💡 Try reducing image complexity or using a simpler preset."
                                )
                                st.info(
                                    "💡 Check your Kaggle backend logs for memory issues"
                                )
                                st.stop()

                        except requests.exceptions.ConnectionError as e:
                            if attempt < max_retries - 1:
                                st.warning(
                                    f"🔌 Attempt {attempt + 1} failed due to connection error: {str(e)}"
                                )
                                st.info(
                                    "💡 This usually means the backend crashed or ran out of memory"
                                )
                                st.info(
                                    "💡 Check your Kaggle backend for error messages"
                                )
                                time.sleep(3)  # Wait longer for connection issues
                                continue
                            else:
                                st.error(
                                    "🔌 Connection error after all retries. Please check if the backend is running."
                                )
                                st.error(f"🔍 Error details: {str(e)}")
                                st.info(
                                    "💡 Ensure your ngrok URL is up to date and backend is accessible."
                                )
                                st.info(
                                    "💡 Check Kaggle backend logs for crashes or memory issues"
                                )
                                st.stop()

                        except Exception as e:
                            if attempt < max_retries - 1:
                                st.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                                st.info("💡 Unexpected error occurred, retrying...")
                                time.sleep(2)
                                continue
                            else:
                                st.error(f"❌ All attempts failed with error: {str(e)}")
                                st.info(
                                    "💡 Check your Kaggle backend logs for detailed error information"
                                )
                                st.stop()

                except Exception as e:
                    st.error(f"❌ Retry logic failed: {str(e)}")
                    st.stop()

                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        # Convert base64 image back to PIL
                        image_data = base64.b64decode(result["image"])
                        result_image = Image.open(io.BytesIO(image_data))

                        # Show success message
                        if (
                            detected_style == "realistic"
                            and "Creative"
                            in st.session_state.get("preset_choice", "Default")
                        ):
                            st.success(
                                f"✅ {sketch_type.title()} realistic generation successful!"
                            )
                            st.info(
                                f"🎯 Using {control_type.title()} ControlNet for optimal results"
                            )
                        else:
                            st.success(
                                f"✅ {sketch_type.title()} {detected_style} generation successful!"
                            )

                        st.info(
                            f"📊 Used optimal parameters: ControlNet={params['controlnet_conditioning_scale']}, Guidance={params['guidance_scale']}"
                        )

                        # Add download button
                        buf = io.BytesIO()
                        result_image.save(buf, format="PNG")
                        st.download_button(
                            label="💾 Download Image",
                            data=buf.getvalue(),
                            file_name=f"{detected_style}_{sketch_type}_{int(time.time())}.png",
                            mime="image/png",
                        )
                        # Display the result
                        st.image(
                            result_image,
                            caption=f"SDXL + ControlNet {detected_style.title()} Result",
                            use_container_width=True,
                        )
                        # Show curtain reveal
                        show_curtain_reveal(result_image)
                    else:
                        st.error(
                            f"❌ Backend error: {result.get('detail', 'Unknown error')}"
                        )
                        st.stop()
                else:
                    st.error(f"❌ Backend error: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

        elif model == "Linoyts Approach (Kaggle)":
            # Use Linoyts Approach API (exact linoyts/sketch-to-3d implementation)
            st.info("🎨 Using Linoyts Approach for optimal sketch adherence...")

            # For Linoyts approach, use the theme as prompt if no custom prompt is provided
            linoyts_prompt = prompt if prompt else theme
            result_image = call_linoyts_approach_api(
                img_bytes, linoyts_prompt, negative_prompt
            )

            if result_image:
                # Reveal
                show_curtain_reveal(result_image)

                # Display the result
                st.image(
                    result_image, caption="Generated Image", use_container_width=True
                )

                # Download button
                img_buffer = io.BytesIO()
                result_image.save(img_buffer, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=img_buffer.getvalue(),
                    file_name=f"linoyts_generated_{int(time.time())}.png",
                    mime="image/png",
                )
            else:
                st.error("❌ Failed to generate image with Linoyts Approach")
                st.stop()

        elif model == "3D Model (HF)":
            # Use Hugging Face 3D Model API
            st.info("🎨 Using Hugging Face 3D Model API for advanced 3D generation...")

            # For 3D, use the theme as prompt if no custom prompt is provided
            sdxl_prompt = prompt if prompt else theme
            result_3d = call_3d_model_api(img_bytes, sdxl_prompt, negative_prompt)

            if result_3d and result_3d["type"] == "3d":
                # Reveal
                show_curtain_reveal(result_image)

                # Display the result
                st.image(
                    result_image,
                    caption="Your 3D masterpiece is ready!",
                    use_container_width=True,
                )

                # Show info about 3D model
                st.success("🎨 3D Model Generated Successfully!")
                st.info(
                    "💡 This 3D model can be used in game engines, 3D software, or VR applications!"
                )

            else:
                st.error("❌ Failed to generate 3D model")
                st.stop()

        elif model == "SDXL ControlNet (HF)":
            # Use Hugging Face API instead of local backend
            st.info("🎨 Using Hugging Face SDXL ControlNet for professional quality...")

            # For SDXL, use the theme as prompt if no custom prompt is provided
            sdxl_prompt = prompt if prompt else theme
            result_image = call_hf_sdxl_api(img_bytes, sdxl_prompt, negative_prompt)

            if result_image:
                # Reveal
                show_curtain_reveal(result_image)

                # Display the result
                st.image(
                    result_image,
                    caption="🎨 Generated with SDXL ControlNet",
                    use_column_width=True,
                )

                # Download button
                img_buffer = io.BytesIO()
                result_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                st.download_button(
                    label="📥 Download Generated Image",
                    data=img_buffer.getvalue(),
                    file_name=f"sdxl_generated_{int(time.time())}.png",
                    mime="image/png",
                )
            else:
                st.error("❌ Failed to generate image with SDXL ControlNet")
                st.stop()

        elif model == "SDXL ControlNet (Kaggle)":
            # Use Kaggle SDXL API (highest quality)
            st.info("🎨 Using Kaggle SDXL ControlNet for maximum quality...")

            # For SDXL, use the theme as prompt if no custom prompt is provided
            sdxl_prompt = prompt if prompt else theme
            result_image = call_kaggle_sdxl_api(img_bytes, sdxl_prompt, negative_prompt)

            if result_image:
                # Reveal
                show_curtain_reveal(result_image)

                # Display the result
                st.image(
                    result_image,
                    caption="🎨 Generated with Kaggle SDXL ControlNet",
                    use_column_width=True,
                )

                # Download button
                img_buffer = io.BytesIO()
                result_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                st.download_button(
                    label="📥 Download Generated Image",
                    data=img_buffer.getvalue(),
                    file_name=f"kaggle_sdxl_generated_{int(time.time())}.png",
                    mime="image/png",
                )
            else:
                st.error("❌ Failed to generate image with Kaggle SDXL ControlNet")
                st.stop()

        elif model == "ControlNet":
            # Get backend URL from global configuration
            backend_url = get_backend_url()
        else:
            backend_url = "http://localhost:8000/generate"
        try:
            response = requests.post(backend_url, files=files, data=data, timeout=600)

            if response.status_code == 200:
                try:
                    result_img = bytes_to_image(response.content)
                except Exception as img_err:
                    st.error(f"Failed to decode image: {img_err}")
                    st.info(f"Backend returned: {response.content[:200]}")
                    st.stop()

                # Reveal
                show_curtain_reveal(result_img)

                # Display the result
                st.image(
                    result_img,
                    caption="Your masterpiece is ready!",
                    use_container_width=True,
                )

                # Download button
                img_buffer = io.BytesIO()
                result_img.save(img_buffer, format="PNG")
                st.download_button(
                    "Download Painting",
                    img_buffer.getvalue(),
                    file_name="painting.png",
                    mime="image/png",
                )

                st.success("🎉 Your masterpiece is ready!")

        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
        except Exception as e:
            st.error(f"Failed to generate painting: {e}")

    # Check if we have canvas data and should show the main interface
    if st.session_state.get("canvas_data") is not None or uploaded_file is not None:
        # Main interface content would go here
        pass
    else:
        st.info("Draw something or upload an image to get started!")


# Add token counting helper function
def count_tokens_approximate(text):
    """Approximate token count (rough estimate: 1 token ≈ 4 characters)"""
    return len(text.split()) + len(text) // 4


def check_prompt_length(prompt, max_tokens=77):
    """Check if prompt is approaching token limit and warn user"""
    token_count = count_tokens_approximate(prompt)
    if token_count > max_tokens:
        st.warning(
            f"⚠️ Your prompt is {token_count} tokens (limit: {max_tokens}). Some details may be cut off!"
        )
        st.info(
            "💡 Consider shortening your prompt to prioritize key details like hair color, clothing, etc."
        )
        return False
    elif token_count > max_tokens * 0.8:  # Warning at 80% of limit
        st.info(
            f"ℹ️ Your prompt is {token_count}/{max_tokens} tokens. Consider shortening to avoid truncation."
        )
        return True
    return True


# Add Qwen2.5-VL integration for smart sketch analysis
import openai

# Configure OpenRouter API for Qwen2.5-VL
OPENROUTER_API_KEY = (
    "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"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Auto-generation trigger
if st.session_state.get("auto_generate_now") and st.session_state.get("user_prompt"):
    st.sidebar.success("🚀 Auto-generating painting with AI prompt...")
    # Reset the flag
    st.session_state.auto_generate_now = False
    # The generation will happen in the main logic below

# Auto-generation button for AI analysis results
if st.session_state.get("last_analysis") and st.session_state.get("user_prompt"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Analysis Complete!")

    # Show analysis summary
    analysis = st.session_state.last_analysis
    if "person_details" in analysis:
        gender = analysis["person_details"].get("gender", "unknown")
        age = analysis["person_details"].get("age_group", "unknown")
        pose = analysis["person_details"].get("pose", "unknown")
        st.sidebar.info(f"👤 **Detected**: {gender.title()} {age}, {pose} pose")

    if "generated_prompt" in analysis:
        st.sidebar.success(f"📝 **AI Prompt**: {analysis['generated_prompt'][:100]}...")

    # Auto-generation button
    if st.sidebar.button(
        "🎨 Generate Painting Now", type="primary", key="auto_generate_btn"
    ):
        st.session_state.trigger_generation = True
        st.sidebar.success("🚀 Starting generation with AI prompt...")
        st.rerun()
