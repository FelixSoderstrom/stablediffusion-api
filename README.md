# Important
**We need to run 3.10 here!**
We decided to go with python version 3.10 beause there was a compability issue with PyTorch and 3.13.
3.10 was the most stable version we could do with all of the packages working together.
Alot of the packages listed in requirements have been downgraded to ensure 3.10 compability.

# Setup
```bash
python-3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
# Before you run:
```
- Downloaded and place your SDXL-model in the models-directory.

- We are using the JuggernautXL Lightning model for testing.
Additional tweaks might be needed if you opt for a different one.

- Make sure to add the exact filename of your model to an .env-file under the variable name 'MODEL_NAME' (without the .safetensors).
```

# Run
```bash
python test.py
```

# Current state
We have a working image generator using the JuggernautXL model in Stable Diffusion. Run test.py to open an image.
Images are now 1024x1024. This is significantly larger than the old code.
But since we are going to run this on AWS we will be fine.

The class no longer enhances prompts. This will have to be done from whereever we use this (soon to be) API.
This means: prompt in, image out.

No API functionality implemented, this is purely local at the moment.
Will start implementing API later today/next week.

# Next step
Implement the API
This whole thing should work as an API where we send prompts here and return images.
This enables us to use this in more places than just AdventureAI.
The plan is to run the API on AWS alongside the Mistral model and have 2 separate API's for the game. More on that in the backend repo.

- Need to look into how to send images over API.
I understand that pillow objects is a no go lol.
That kind of only leaves b64 encoding. Will do more research during lunch.
