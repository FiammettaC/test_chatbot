from flask import Flask, render_template, request
from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM, FalconForCausalLM
from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np
import torch

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(question):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")    
    encoded_input = tokenizer([question],
                                return_tensors='pt',
                                max_length=1024,
                                truncation=False).to(device)
    model = model.to(device)
    output = model.generate(input_ids = encoded_input.input_ids,
                            attention_mask = encoded_input.attention_mask,
                            num_beams=5,
                            max_new_tokens=512,
                            penalty_alpha=0.6, 
                            top_k=4,
                            do_sample=False,
                            )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

@app.route("/")
def home():    
    return render_template("index.html")

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    response = predict(userText)  
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)