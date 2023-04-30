from transformers import pipeline, BartForConditionalGeneration , AutoTokenizer
from flask import Flask, request, jsonify

## Options
#./model/bart-large-cnn
#./model/distilbart-cnn-12-6
llm_model = './model/distilbart-cnn-12-6'

model = BartForConditionalGeneration.from_pretrained(llm_model)
tokenizer = AutoTokenizer.from_pretrained(llm_model)
device=0

text_example = """The tower is 324 meters (1,063 ft) tall, about the same height as an 81-storey building, 
and the tallest structure in Paris. Its base is square, measuring 125 meters (410 ft) on each side. 
During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure 
in the world, a title it held for 41 years until the Chrysler Building in New York
City was finished in 1930. It was the first structure to reach a height of 300 meters. 
Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is 
now taller than the Chrysler Building by 5.2 meters (17 ft). Excluding transmitters, 
the Eiffel Tower is the second tallest free-standing structure in France
after the Millau Viaduct."""

summarizer = pipeline(
    "summarization", 
    model = model,
    tokenizer = tokenizer,
    device = device
)

print(summarizer(text_example, max_length=130, min_length=30, do_sample=False))