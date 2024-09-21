import gradio as gr

from classifier import AnimalClassifier

classifier = AnimalClassifier()

with gr.Blocks() as page:
    gr.Markdown("""
    # Animals Classifier
    Upload an image of an animal and the AI will try to classify it
    """)
    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload an image", width=600, height=600)
        output = gr.Markdown("## Result")
    classify_btn = gr.Button("Classify")
    classify_btn.click(fn=classifier.classify, inputs=image_input, outputs=output)
    image_input.clear(fn=lambda: "## Result", inputs=[], outputs=output)

if __name__ == "__main__":
    page.launch()
