from transformers import pipeline, BartForConditionalGeneration , AutoTokenizer
from flask import Flask, request, jsonify
import torch

"""
## Tested options for your LLMs
> distilbart-cnn-12-6
    - supports truncation, max_length
> bart-large-cnn
    - does not supports truncation, max_length
    - possible better results.
## devices
>  device=1,     # to utilize GPU cuda:1
>  device=0,     # to utilize GPU cuda:0
>  device=-1)    # default value to utilize CPU
"""
llm_model = './model/distilbart-cnn-12-6'
model = BartForConditionalGeneration.from_pretrained(llm_model)
tokenizer = AutoTokenizer.from_pretrained(llm_model)
device=0

summarizer = pipeline(
    "summarization", 
    model = model,
    tokenizer = tokenizer,
    device = device
)

def summ(text: str):
    return summarizer(text, max_length=512, min_length=30, do_sample=False, truncation=True) 

# flask server
app = Flask(__name__)

# define routes
@app.route("/", methods=["POST"])
def index():
    if request.method == "POST":
        data = request.json

        if data is None:
            return jsonify(
                {   
                    "error": "Invalid JSON request"
                }
            )
        elif not ("text" in data):
            return jsonify(
                {
                    "error": "'text' field(s) not present in JSON request"
                }
            )
        elif not (
            isinstance(data["text"], str)
        ):
            return jsonify(
                {
                    "error": "'text' field is not a string"
                }
            )
        try:
            result = summ(data["text"])
            return jsonify(result)
        except Exception as exception:
            return jsonify({"error": str(exception)})

#print(summarizer(text_example, max_length=130, min_length=30, do_sample=False))

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
