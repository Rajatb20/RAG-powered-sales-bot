import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from sales_gpt.agents import SalesGPT

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "azure")

def initialize_agent():
    if not API_KEY or not BASE_URL:
        return None

    llm = AzureChatOpenAI(
        azure_endpoint=BASE_URL,
        openai_api_version=OPENAI_API_VERSION,
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        openai_api_type=OPENAI_API_TYPE,
        temperature=0.1,
    )

    config = dict(
        consultant_name="AI Assistant",
        consultant_role="Exercise Product Consultant",
        company_name="Peloton",
        company_business="Peloton is the leading provider of exercise accessories, or machines delivering unrivaled outputs, robust technology and expert services to fuel customers’ success through all phases of lifecycle.",
        company_values="We aim to empower our clients to achieve excellence in health by providing unparalleled machines and accessories, services throughout every stage of the lifecycle.",
        conversation_purpose="Collect user specifications pertaining to their desired exercise product, consolidate them, and suggest the best suited product.",
        conversation_history=[],
        conversation_type="call",
        use_tools=True,
    )

    sales_agent = SalesGPT.from_llm(llm, verbose=True, **config)
    sales_agent.seed_agent()
    return sales_agent

# Initialize Global Agent
try:
    sales_agent = initialize_agent()
except Exception as e:
    print(f"Agent init error: {e}")
    sales_agent = None

def sales_bot_chat(message, history):
    if not sales_agent:
        return "⚠️ **System Error**: Agent not initialized. Please check your `.env` configuration (API_KEY, BASE_URL, etc.)."

    # Process user input
    sales_agent.human_step(message)
    
    # Generate response
    try:
        sales_agent.step()
    except Exception as e:
        return f"⚠️ **Error during agent step**: {str(e)}"

    # Extract response from history
    last_msg = sales_agent.conversation_history[-1]
    
    # Process the response text to remove prefixes and end tokens
    response_text = last_msg
    if ":" in response_text:
        # Split by first colon and take the rest
        response_text = response_text.split(":", 1)[1].strip()
    
    response_text = response_text.replace("<END_OF_TURN>", "").strip()
    
    return response_text

# Create a beautiful theme using Gradio's native options
# Peloton colors are typically Black, White, and Red (#df1c2f)
peloton_theme = gr.themes.Soft(
    primary_hue="red",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Montserrat"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    block_border_width="1px",
    block_title_text_weight="600",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)

# Custom header using Markdown instead of HTML
header_markdown = """
# 🚴 Peloton AI
### Your Professional Fitness Product Consultant
---
Welcome! I'm here to help you find the perfect Peloton equipment for your fitness journey.
"""

# Build the Gradio App
with gr.Blocks(theme=peloton_theme, title="Peloton Sales Assistant") as demo:
    gr.Markdown(header_markdown)
    
    chat_interface = gr.ChatInterface(
        fn=sales_bot_chat,
        examples=[
            "Tell me about the Peloton Bike+.",
            "I'm looking for a treadmill for marathon training.",
            "What accessories do you recommend for a beginner?",
            "What are the benefits of the Peloton Row?"
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
