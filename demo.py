import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import re

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = 'ufal/robeczech-base'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
transformers.logging.set_verbosity(transformers.logging.ERROR)

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def classify_sentence(sentence:str):
    sentence = tokenizer(sentence,truncation=True,return_tensors="pt")
    model.eval()
    with torch.no_grad():
        sentence.to(device)
        output = model(**sentence)
    
    classification = F.softmax(output.logits,dim=1)[0]
    return classification

def classify_sentence_wrapper(sentence:str):
    result = classify_sentence(sentence)
    return {'Neutral':result[0].item(),'Female':result[1].item(),'Male':result[2].item()}

def classify_text(text:str):
    text = split_into_sentences(text)
    result = list(map(classify_sentence,text))
    result = np.array([t.numpy() for t in result]).argmax(axis=1)
    return result

def classify_text_wrapper(text:str):
    result = classify_text(text)
    n = len(result)
    no_male = np.where(result==2)[0].shape[0]
    no_female = np.where(result==1)[0].shape[0]
    no_neutral = np.where(result==0)[0].shape[0]

    return {'Neutral':no_neutral/n,'Female':no_female/n,'Male':no_male/n}

def interpret_gender(text:str):
    result = classify_text(text)
    split = split_into_sentences(text)
    interpretation = []
    
    for idx,sentence in enumerate(split):
        score = 0
        if result[idx] == 1:
            score = 1
        if result[idx] == 2:
            score = -1
        interpretation.append((sentence, score))
        
    return interpretation


if __name__ == "__main__":
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained("sagittariusA/gender_classifier_cs")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    print('Model loaded.')

    with open('./data/example.txt', 'r') as file:
        txt_sample = [[file.read()]]

    #Gradio interface config
    label = gr.outputs.Label(num_top_classes=3)
    inputs = gr.inputs.Textbox(placeholder=None, default="", label=None)
    app = gr.Interface(fn=classify_text_wrapper,title='Gender bias classifier',theme='default',
                    inputs="textbox",layout='unaligned', outputs=label, capture_session=True,
                    interpretation=interpret_gender,examples=txt_sample)

    app.launch(inbrowser=True)
