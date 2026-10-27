"""Gradio web interface for the Icebreaker Bot (Local Ollama Edition)."""

import sys
import io
import logging
import uuid

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gradio as gr

from modules.data_extraction import (
    extract_linkedin_profile,
    list_available_profiles,
    get_profile_label_map,
)
from modules.data_processing import (
    split_profile_data,
    create_vector_database,
    verify_embeddings,
)
from modules.llm_interface import change_llm_model
from modules.query_engine import (
    generate_initial_facts,
    answer_user_query,
)

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ],
)

logger = logging.getLogger(__name__)

# Dictionary to store active conversations: session_id -> {index, profile_data}
active_sessions = {}


def process_profile(selected_label, selected_model):
    """
    Process a LinkedIn profile and generate initial facts.
    """
    try:
        if selected_model and selected_model.strip() != config.LLM_MODEL_ID:
            change_llm_model(selected_model.strip())

        profile_data = extract_linkedin_profile(selected_label)

        if not profile_data:
            return (
                "⚠️ Failed to retrieve profile data. Please check the profile selection.",
                None,
                gr.update(visible=False),
            )

        nodes = split_profile_data(profile_data)

        if not nodes:
            return "⚠️ Failed to process profile data into text nodes.", None, gr.update(visible=False)

        index = create_vector_database(nodes)

        if not index:
            return "⚠️ Failed to create vector database.", None, gr.update(visible=False)

        if not verify_embeddings(index):
            logger.warning("Some embeddings may be missing or invalid")

        facts = generate_initial_facts(index)

        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "index": index,
            "profile": profile_data,
        }

        name = profile_data.get("full_name", "Unknown")
        headline = profile_data.get("headline", "")
        city = profile_data.get("city", "")
        country = profile_data.get("country_full_name", "")
        location_str = f"📍 {city}, {country}" if city and country else ""
        skills = profile_data.get("skills", [])
        skills_str = " • ".join(skills[:8]) if skills else "N/A"

        header_card = f"""### 👤 {name}
**{headline}**  
{location_str}  

> **Core Competencies:** `{skills_str}`
"""

        result_markdown = f"""{header_card}

---

### 💡 Icebreaker Facts
{facts}
"""

        return result_markdown, session_id, gr.update(visible=True)

    except Exception as e:
        logger.error(f"Error in process_profile: {e}")
        return f"❌ Error: {str(e)}", None, gr.update(visible=False)


def chat_with_profile(session_id, user_query, chat_history):
    """
    Chat with a processed LinkedIn profile.
    """
    if not session_id or session_id not in active_sessions:
        return chat_history + [
            [user_query, "⚠️ No active profile session found. Please click **'⚡ Generate Icebreakers'** first!"]
        ]

    if not user_query or not user_query.strip():
        return chat_history

    try:
        session = active_sessions[session_id]
        index = session["index"]

        response = answer_user_query(index, user_query)

        if hasattr(response, "response"):
            bot_text = response.response.strip()
        else:
            bot_text = str(response)

        return chat_history + [[user_query, bot_text]]

    except Exception as e:
        logger.error(f"Error in chat_with_profile: {e}")
        return chat_history + [[user_query, f"❌ Error: {str(e)}"]]


def quick_ask(query_text, session_id, chat_history):
    """Helper to inject sample question clicks directly into chat."""
    return chat_with_profile(session_id, query_text, chat_history)


def create_gradio_interface():
    """Create the polished Gradio interface for the Icebreaker Bot."""

    label_map = get_profile_label_map()
    choices = list(label_map.keys())

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(title="LinkedIn Icebreaker Bot (Local Ollama)", theme=custom_theme) as demo:
        gr.Markdown(
            """
            # 🧊 LinkedIn Icebreaker Bot
            ### 100% Local RAG-Powered Profile Intelligence with **Ollama** & **HuggingFace**
            *Generate personalized conversation starters, extract key accomplishments, and chat with LinkedIn profiles completely offline without cloud API keys.*
            """
        )

        with gr.Tabs():
            with gr.TabItem("🔍 1. Profile Intelligence & Icebreakers"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚙️ Configuration")
                        profile_dropdown = gr.Dropdown(
                            choices=choices,
                            label="Select a LinkedIn Profile",
                            value=choices[0] if choices else None,
                        )

                        model_input = gr.Textbox(
                            label="Local Ollama Model",
                            value=config.LLM_MODEL_ID,
                            info="Model running in your local Ollama daemon (e.g. llama3.2, mistral, deepseek-coder:6.7b)",
                        )

                        process_btn = gr.Button("⚡ Generate Icebreakers", variant="primary", size="lg")

                        gr.Markdown(
                            """
                            ---
                            **💡 How it works:**
                            1. Loads verified profile data from local JSON.
                            2. Splits information into semantic chunks.
                            3. Computes vector embeddings locally (`all-MiniLM-L6-v2`).
                            4. Runs LlamaIndex RAG with **Ollama** to synthesize unique conversation starters.
                            """
                        )

                    with gr.Column(scale=2):
                        result_markdown = gr.Markdown(
                            value="*Select a profile and click **Generate Icebreakers** to extract interesting facts and conversation openers.*",
                        )
                        session_id = gr.Textbox(label="Session ID", visible=False)

            with gr.TabItem("💬 2. Interactive Profile Chat"):
                gr.Markdown("Ask specific questions, draft cold outreach messages, or interview questions based on the selected profile.")

                chatbot = gr.Chatbot(
                    height=450,
                    show_label=False,
                    bubble_full_width=False,
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="e.g. Draft a 2-sentence warm outreach email connecting our shared interest in AI...",
                        label="Ask a question about the profile",
                        scale=9,
                    )
                    send_btn = gr.Button("Send 📨", variant="primary", scale=1)

                gr.Markdown("**Quick Prompts:**")
                with gr.Row():
                    q1 = gr.Button("🎯 What are their biggest achievements?", size="sm")
                    q2 = gr.Button("🎓 Summarize education & early career", size="sm")
                    q3 = gr.Button("✉️ Write a cold outreach LinkedIn message", size="sm")
                    q4 = gr.Button("🤝 What common interests could we bond over?", size="sm")

                clear_btn = gr.ClearButton([chat_input, chatbot], value="Clear Conversation", size="sm")

        # Wiring events
        process_btn.click(
            fn=process_profile,
            inputs=[profile_dropdown, model_input],
            outputs=[result_markdown, session_id],
        )

        send_btn.click(
            fn=chat_with_profile,
            inputs=[session_id, chat_input, chatbot],
            outputs=[chatbot],
        ).then(lambda: "", None, chat_input)

        chat_input.submit(
            fn=chat_with_profile,
            inputs=[session_id, chat_input, chatbot],
            outputs=[chatbot],
        ).then(lambda: "", None, chat_input)

        # Quick question helpers
        q1.click(lambda: "What are their biggest career achievements?", None, chat_input).then(
            fn=chat_with_profile, inputs=[session_id, chat_input, chatbot], outputs=[chatbot]
        ).then(lambda: "", None, chat_input)

        q2.click(lambda: "Summarize their education and early career journey.", None, chat_input).then(
            fn=chat_with_profile, inputs=[session_id, chat_input, chatbot], outputs=[chatbot]
        ).then(lambda: "", None, chat_input)

        q3.click(lambda: "Draft a personalized, professional 3-sentence LinkedIn connection request referencing their work.", None, chat_input).then(
            fn=chat_with_profile, inputs=[session_id, chat_input, chatbot], outputs=[chatbot]
        ).then(lambda: "", None, chat_input)

        q4.click(lambda: "What volunteer work, publications, or personal interests could be great conversation starters?", None, chat_input).then(
            fn=chat_with_profile, inputs=[session_id, chat_input, chatbot], outputs=[chatbot]
        ).then(lambda: "", None, chat_input)

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=5000,
    )
