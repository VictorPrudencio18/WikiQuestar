# WikiQuestar | An Answer-Aware Question Generation Application Using Wikipedia as the Knowledge Source
**WikiQuestar** is a groundbreaking system that leverages the potential of machine learning and Wikipedia to **generate thought-provoking questions**. This system employs advanced natural language processing techniques to **extract valuable information from Wikipedia articles**, thereby enhancing the quality of generated questions.

The project is designed to **enhance learning outcomes**, **foster comprehension**, and **broaden knowledge acquisition**. It offers users a dynamic platform for generating questions that are tailored to specific topics or areas of interest. Whether employed in educational settings, content creation, or research, WikiQuestar serves as a powerful tool that stimulates critical thinking and deepens understanding.

# Tools Used
WikiQuestar is implemented using the following Python packages:

| Package | Description |
| --- | --- |
| Wikipedia | A Wikipedia API to make it my knowledge source for Retrieval Augmented Generation (RAG)  |
| PyTorch | An Open-source machine learning framework |
| Transformers | A Hugging Face package contains state-of-the-art Natural Language Processing models |
| Datasets | A Hugging Face package contains popular open-source datasets |
| Evaluate | A Hugging Face package contains several evaluation metrics like BLEU, ROUGE, METEOR, BERTScore, etc. |
| Streamlit | A popular data deployment Python package |

# Usage
## Running Demo:
https://github.com/MohammedAly22/WikiQuestar/assets/90681796/b55d3b72-d3dc-4bcd-a01d-91a2d7327c6a

## Usage as a high-level Pipeline:
1. Define some useful functions for highlighting the answer in the paragraph and preparing the instruction prompt that will be fed to the model: 
```Python
def highlight_answer(context, answer):
    context_splits = context.split(answer)
    
    text = ""
    for split in context_splits:
        text += split
        text += ' <h> '
        text += answer
        text += ' <h> '
        text += split
    
    return text
```
```Python
def prepare_instruction(answer_highlighted_context):
    instruction_prompt = f"""Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks.
    context:
    ```
    {answer_highlighted_context}
    ```
    """
    
    return instruction_prompt
```

2. Use the model as a Hugging Face Pipeline:
```Python
from transformers import pipeline

pipe = pipeline('text2text-generation', model='mohammedaly2222002/t5-small-squad-qg-v2')

context = """During the 2011–12 season, he set the La Liga and European records\
for most goals scored in a single season, while establishing himself as Barcelona's\
all-time top scorer. The following two seasons, Messi finished second for the Ballon\
d'Or behind Cristiano Ronaldo (his perceived career rival), before regaining his best\
form during the 2014–15 campaign, becoming the all-time top scorer in La Liga and \
leading Barcelona to a historic second treble, after which he was awarded a fifth \
Ballon d'Or in 2015. Messi assumed captaincy of Barcelona in 2018, and won a record \
sixth Ballon d'Or in 2019. Out of contract, he signed for French club Paris Saint-Germain\
in August 2021, spending two seasons at the club and winning Ligue 1 twice. Messi \
joined American club Inter Miami in July 2023, winning the Leagues Cup in August of that year.
"""

answer_highlighted_context = highlight_answer(context=context, answer='Inter Miami')
prompt = prepare_instruction(answer_highlighted_context)
```

This will be the final prompt:
```
Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks
context:
```During the 2011–12 season, he set the La Liga and European records\
for most goals scored in a single season, while establishing himself as Barcelona's\
all-time top scorer. The following two seasons, Messi finished second for the Ballon\
d'Or behind Cristiano Ronaldo (his perceived career rival), before regaining his best\
form during the 2014–15 campaign, becoming the all-time top scorer in La Liga and \
leading Barcelona to a historic second treble, after which he was awarded a fifth \
Ballon d'Or in 2015. Messi assumed captaincy of Barcelona in 2018, and won a record\
 sixth Ballon d'Or in 2019. Out of contract, he signed for French club Paris Saint-Germain\
in August 2021, spending two seasons at the club and winning Ligue 1 twice. Messi \
joined American club  <h> Inter Miami <h> in July 2023, winning the Leagues Cup in August of that year.```
```

3. Use the loaded `pipeline` to generate questions their answer is `Inter Miami`:
```Python
outputs = pipe(prompt, num_return_sequences=3, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)
for output in outputs:
    print(output['generated_text'])
```

Result:
```
1. What club did Messi join in the 2023 season?
2. What was Messi's name of the club that won the Leagues Cup on July 20?
3. What club did Messi join in the Leagues Cup in July 2023?
```

# Dataset
**The Stanford Question Answering Dataset (SQuAD)** is a benchmark dataset in the field of natural language processing and machine reading comprehension, developed by researchers at Stanford University. SQuAD is a collection of real questions posed by a group of crowd workers on a set of Wikipedia articles, each paired with a corresponding passage from the article. The answers to these questions are segments of text from the corresponding passage.

The SQuAD dataset is designed to train and evaluate machine learning models to comprehend and answer questions in natural language. It has been used as a benchmark for evaluating the performance of various question-answering systems and models, including both rule-based systems and deep learning-based approaches, such as neural network models.

# Methodology
## Dataset Preparation
1. In my research, I incorporated a highlight token `<h>` into the context `c` to better highlight the answer `a`. This was done following the work of Chan and Fan (2019), where the token `<h>` was introduced to represent the answer within the context. The resulting sequence `x` is represented as follows:

$x = [ c_1, ..., \lt h\gt , a_1, ..., a_a, \lt h\gt , ..., c_c ]$

This method allows for a more accurate understanding of the context and the answer, enhancing the performance of the machine learning model.

2. Preparing the instruction prompt by following this template:
```
Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks.
context:
```{answer_highlighted_context}```
```

# Results
I've done full fine-tuning on the `t5-small` model with the following configuration hyperparameters:

| Model | epochs | batch size | warmup steps | weight decay | gradient accumulation steps | learning rate | save total limit | fp16 | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T5-Small-FFT | 10 | 16 | 1000 | 0.01 | 4 | 5e-5 | 2 | True |

Model results:
| Model           | BLEU | Rouge1 | Rouge2 | RougeL | RougeLSum | METEOR | BertScore | 
| ---             | ---  | ---    | ---    | ---    | ---       | ---    | ---       |
| T5-Small-FFT | 20.00 | 47.69  | 26.43  | 44.15  | 44.15     | 45.84  | 91.82     |
