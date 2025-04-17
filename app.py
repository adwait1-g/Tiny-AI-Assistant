# Using gradio for the assistant UI

import gradio as gr
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# The pre-trained model is always present at "./model".
model_path = "./model"

# Get the generator
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def generate_reply (prompt):
  response = generator(prompt, max_length=100, do_sample=True, top_p=0.95, top_k=50)
  gen_text = response[0]['generated_text']
  gen_text = gen_text.replace(prompt, '')
  return gen_text

interface = gr.Interface(fn=generate_reply,
                         inputs = gr.Textbox(lines=4, placeholder="Ask me anything..."),
                         outputs = gr.Textbox(),
                         title="AIssistant",
                         description="A lightweight assistant powered by tiny-gpt2")

# Launch app
interface.launch(server_name="0.0.0.0", server_port=8080)