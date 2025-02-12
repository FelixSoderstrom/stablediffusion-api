# Important
**Please ensure you are using python 3.10!**
To run this script and create your virtual environment, make sure to use Python 3.10.

# Setup
To set up your environment, execute the following commands in your terminal:
```bash
python-3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Before you run:
```
- Download and place your SDXL model in the `models` directory.
- We are currently using the JuggernautXL Lightning model for testing. Additional adjustments may be necessary if you choose a different model.
- Ensure you add the exact filename of your model to an `.env` file under the variable name `MODEL_NAME` (without the `.safetensors` extension).
```

# To generate a single image:
```bash
python test.py
```

# Using the web UI:
```bash
uvicorn src.api:app --reload
```
Then, open the `webapp.html` file in your browser and enter your prompt.

# Current state
We are now using a persistent endpoint that reuses the pipeline initialization.
The pipeline is initialized when an IP pings the endpoint.
The pipeline is then cleaned after 5 minutes of inactivity.
If the same IP pings the endpoint within these 5 minutes, the pipeline is reused.

This allows the used to generate images quicker but more importantly;
It enables multiple users to generate images simultaneously.

By default the pipeline is not built to be used to generate more than one image at a time.
Giving each user an instance of the class solves this problem.

When everything works like it should, it takes a while to initialize the pipeline but then image generation is done very fast.

# Next step
- Discuss the use of the .env
We are currently using .env to specify which model to use.
In the previous project we had more environment variables so this made sense.
Hardcoding the model name is not a good solution.
We could add it as an argument to the StableDiffusion class, initially provided by the endpoint (who cares if its hardcoded in our webapp for the game??)