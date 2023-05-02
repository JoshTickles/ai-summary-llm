from transformers import pipeline, BartForConditionalGeneration , AutoTokenizer
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

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
device=-1

summarizer = pipeline(
    "summarization", 
    model = model,
    tokenizer = tokenizer,
    device = device
)

def summ(text: str):
    return summarizer(text, max_length=512, min_length=30, do_sample=False, truncation=True) 

class Data(BaseModel):
    text: str

# setup api
app = FastAPI()

# define routes
@app.get("/")
def home():
    return {"message": "API Ready"}
@app.post("/")
async def home_post(data : Data):
        if data is None:
            return (
                {
                    "error": "Invalid JSON request"
                }
            )
        elif not (
            isinstance(data.text, str)
        ):
            return (
                {
                     "error": "'text' field is not a string"
                }
            )
        try:
            result = summ(data.text)
            return (result)
        except Exception as exception:
            return ({"error": str(exception)})

#print(summarizer(text_example, max_length=130, min_length=30, do_sample=False))

if __name__ == "__main__":
    uvicorn.run("app-fastapi:app", port=5000, reload=True)