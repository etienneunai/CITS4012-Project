# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=5.0.1",
#     "gdown>=6.3.0",
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import gdown
    import os

    if not(os.path.exists('MultiRC/train_456-fixedIds.json') and os.path.exists('MultiRC/test_83-fixedIds.json')):
        gdown.download_folder(id="18SlXjdkhUrG_PE0aTysMypMKx41a9evH")

    train_json = 'MultiRC/train_456-fixedIds.json'
    test_json = 'MultiRC/test_83-fixedIds.json'


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Readme
    *If there is something to be noted for the marker, please mention here.*

    *If you are planning to implement a program with Object Oriented Programming style, please put those the bottom of this ipynb file*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1.Dataset Processing
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _():
    import json
    with open(train_json, 'r') as f:
        train_data = json.load(f)['data']
    with open(test_json, 'r') as f:
        test_data = json.load(f)['data']

    print(train_data[1])
    return (train_data,)


@app.cell
def _(train_data):
    import re

    def clean_passage(text):
        # Remove all html tags and sentence labels
        text = re.sub(r"<b>\s*Sent\s*\d+\s*:\s*</b>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text).strip()  
        return text

    def flatten_data(data):
        # Converts data into a list of tuples (passage, question, answer, label)
        flattened_data = []
        for item in data:
            paragraph = item['paragraph']
            # Tidy Passage
            passage = clean_passage(paragraph['text'])

            for q in paragraph['questions']:
                question_text = q['question']
                for a in q['answers']:
                    answer_text = a['text']
                    answer_val= a['isAnswer']
                    flattened_data.append((passage,question_text,answer_text,answer_val))
        
        return flattened_data

    with open("output.txt", "a", encoding="utf-8") as fle:      
        flat_train = flatten_data(train_data)                                                                                              
        for i in flat_train:

            fle.write(str(i)) 


    #flatten_data(test_data)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Model Implementation
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3.Testing and Evaluation
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
