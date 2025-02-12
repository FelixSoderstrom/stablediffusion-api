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
We have one endpoint set up.
It initializes the Stablediffusion pipeline for each request.
I did this as it felt easier to understand and decided to keep it because it might be useable.
For the game we will probbably want to have a persistent pipeline across multiple requests.

# Next step
- Implement a persistent pipeline across multiple requests.

- Discuss the use of the .env
We are currently using .env to specify which model to use.
In the previous project we had more environment variables so this made sense.
Hardcoding the model name is not a good solution.
We could add it as an argument to the StableDiffusion class, initially provided by the endpoint (who cares if its hardcoded in our webapp for the game??)