

from guidance import models, gen

# this relies on the environment variable OPENAI_API_KEY being set
model = models.Transformers('meta-llama/Llama-3.1-8B-Instruct')

lm = model
with instruction():
    lm += "What is a popular flavor?"
lm += gen('flavor', max_tokens=10, stop=".")

print(lm)
