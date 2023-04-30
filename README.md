## Introduction
This is a test project for playing with LLMs to provide text summaries. It's a simple project using transformers. 

You can use either your GPU, or your CPU. Right now I've only got Nvidia devices, so this is tested on that. 

##### Tested options for your LLMs
- distilbart-cnn-12-6
    - supports truncation, max_length
- bart-large-cnn
    - does not supports truncation, max_length
    - possible better results.

##### devices
- device=1, # to utilize GPU cuda:1
- device=0, # to utilize GPU cuda:0
- device=-1 # default value to utilize CPU

## Setup 

Create a Python virtual environment `venv`
```
python3 -m venv venv
```

Activate the virtual environment `py3env`
```
source venv/bin/activate
```

Install the Python packages from `requirements.txt`
```
pip3 install -r requirements.txt
```

Download the model - Note, you may need git-lfs for unpacking the files. 
```
git clone https://huggingface.co/sshleifer/distilbart-cnn-12-6 ./model/distilbart-cnn-12-6
```

In each script, add your variables as required. You basically just need to set the device, and model location. 

Run the app, or the api version of the app.
```
python3 ./app-api.py
```

## Usage for the API

### Request
Send JSON POST requests to `http://<address>:5000`
```json
{
	"text": "When Cyclone Gabrielle hit New Zealand, it left a mark on us beyond the physical destruction.  In this series of short films, made with the support of NZ On Air, four directors confront the deeper impact on four of our worst-affected communities - Esk Valley in Hawke’s Bay, State Highway 35 around East Cape, and Muriwai and Te Henga/Bethells Beach on Auckland’s West Coast. Te Henga residents are mourning not just where they live, but a part of who they are, says Anna Marbrook. “We have experienced the land literally falling away under our feet. ”The River Memory director and Te Henga resident uses the term “ecological grief” to explain it. “It’s a loss of a way of life or a loss of something you thought was going to be, but in fact, it’s not going to be like that in the future. ”Te Henga, or Bethells Beach, is a small and tight-knit community of several hundred people on the west coast of the North Island, near Auckland. Residents are still cleaning up following the destruction from Cyclone Gabrielle on the 13th and 14th of February, but the film describes how the damage goes much deeper. Gabrielle has permanently damaged the landscape in which multiple generations have lived, worked, and made memories. The Waitākere river level rose dramatically during the cyclone, leading to the damage and destruction of bridges and houses throughout the settlement. Marbrook said afterwards she could see great open wounds where sections of hillsides had fallen, tracts of the native bush had been ripped out and tonnes of farmland had quite literally slumped. ",
}
```

### Response
```json
{
	"summary_text": " Four directors confront the deeper impact of Cyclone Gabrielle on four of ourworst-affected communities . Te Henga, or Bethells Beach, is a small and tight-knit community of severalhundred people on the west coast of the North Island, near Auckland . The Waitākere river level rosedramatically during the cyclone, leading to the damage and destruction of bridges and houses ."
}
```
