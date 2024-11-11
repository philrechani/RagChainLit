To run, first download ```mistral-7b-instruct-v0.2.Q2_K.gguf``` from https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main

Create a file ```CONFIG.py``` in the config folder with the following import ```from config.DEFAULT_CONFIG import *```

Then, set all the variables found in ```DEFAULT_CONFIG.py``` in ```CONFIG.py```

Finally, run: ```chainlit run app.py --port 49512```