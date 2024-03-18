import re
import streamlit as st
from utils import pipe, prepare_instruction, get_wikipedia_article, split_content_into_chunks


if 'selected_chunk' not in st.session_state:
    st.session_state['selected_chunk'] = 'chunk 1'
if 'context' not in st.session_state:
    st.session_state['context'] = ''
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = []


def update_context():
    chunk_number = int(re.findall(r'\d+', st.session_state['selected_chunk'])[0])
    if st.session_state['chunks']:
        st.session_state['context'] = st.session_state['chunks'][chunk_number - 1]


# for custom CSS styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Question Generation From Wikipedia')

# Retrieval form
st.markdown('1. Retrieval From')
with st.form('retrieval form'):
    topic = st.text_input('topic', placeholder='Enter a topic to retrieve context about from wikipedia')
    submitted = st.form_submit_button('Retrieve')

    if submitted:
        if topic:
            content, success_code = get_wikipedia_article(topic)
            if success_code == 1:
                st.success(f'Context about "{topic}" Retrieved Successfully from Wikipedia!')
                chunks = split_content_into_chunks(content, 512)
                st.session_state['chunks'] = chunks 
            else:
                st.error(content)
        else:
            st.error('Please, enter a topic to retrieve context about!')


# Choose Different Chunk
selected_chunk = st.selectbox(
    label='Choose Different Chunk',
    options=['chunk ' + str(i+1) for i in range(len(st.session_state['chunks']))],
    key='trigger_update',
    on_change=update_context
)


if st.session_state['chunks']:
    chunk_number = int(re.findall(r'\d+', selected_chunk)[0])
    st.session_state['context'] = st.session_state['chunks'][chunk_number - 1]


# Generation From
st.markdown('2. Generation From')
with st.form('generation form'):
    context = st.text_area(label='Enter Your Context:', placeholder='Please, enter a context to generate question from', height=250, key='context')
    answer = st.text_input(label='Enter Your Answer:', placeholder='Please, copy an answer snippet from the provided context')
    num_of_questions = st.number_input(
        label='Enter a Number of Generated Questions:',
        placeholder='Please, enter a number of generated questions you need',
        min_value=1,
        max_value=5)
    
    submitted = st.form_submit_button('Generate')
    if submitted:
        if context:
            if answer:
                prompt = prepare_instruction(context, answer)
                with st.spinner(f'Generating Questions...'):
                    generated_output = pipe(prompt, num_return_sequences=num_of_questions, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)
            
                st.write('Generated Question(s):')
                for i, item in enumerate(generated_output):
                    st.success(f"Question #{i+1}: {item['generated_text']}")
            else:
                st.error('Please, provide an answer snippet')
        else:
            st.error('Please, provide context or retrieve from the above form')
