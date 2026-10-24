"""Gradio Web Application Entrypoint for WorldQuant Institutional Portfolio Management."""

from src.ui.gradio_app import build_app

if __name__ == "__main__":
    app = build_app()
    custom_css = """
    .gradio-container { background-color: #0E1117 !important; color: #D1D4DC !important; }
    .bk-root { background-color: transparent !important; }
    """
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css)
