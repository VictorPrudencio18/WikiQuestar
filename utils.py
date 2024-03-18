import wikipedia
from transformers import pipeline, AutoTokenizer


pipe = pipeline('text2text-generation', model='mohammedaly2222002/t5-small-squad-qg-v2')
tokenizer = AutoTokenizer.from_pretrained('mohammedaly2222002/t5-small-squad-qg-v2')


def get_wikipedia_article(query):
    try:
        page = wikipedia.page(query)
        return page.content, 1
    
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Did you mean one of these: {', '.join(e.options)}", -1
    
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'.", -2


def prepare_instruction(context, answer):
    context_splits = context.split(answer)
    
    text = ""
    for split in context_splits:
        text += split
        text += ' <h> '
        text += answer
        text += ' <h> '
        text += split
    
    instruction_prompt = f"""Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks.
    context:
    ```
    {text}
    ```
    """
    
    return instruction_prompt


def split_content_into_chunks(content, chunk_size):
    chunks = []
    words = tokenizer.tokenize(content)
    
    while words:
        chunk_tokens = words[:chunk_size]
        chunk = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk)
        words = words[chunk_size:]

    return chunks
