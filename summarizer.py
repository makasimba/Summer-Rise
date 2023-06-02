import openai
import os
from pdfminer.high_level import extract_text

openai.api_key = os.environ['OPENAI_API_KEY']


def summarize(text, model='gpt-3.5-turbo', temperature=0, setting=None):

    setting = [
        {'role': 'system', 'content': """You will be provided with text from a research paper.
                                         Write an eloquent and concise summary of the text in
                                         100-150 words.
                                         """},
        {'role': 'user', 'content': text},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=setting,
        temperature=temperature
    )

    return response.choices[0].message['content']


if __name__ == '__main__':
    paper = 'ml-and-chem.pdf'
    txt = extract_text(paper)
    print('summarizing ... ')
    summary = summarize(txt[:len(txt)//2])
    output = f"""Summary of {paper}:\n\n{summary}"""
    print(output)
