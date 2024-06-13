import os
import numpy as np
import faiss
from transformers import BertTokenizer, BertModel
import torch
import json
import time
import warnings
import copy
import pickle
import random
import torch.nn.functional as F

seed_value = 42 
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


llm_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = BertTokenizer.from_pretrained('facebook/contriever')
model = BertModel.from_pretrained('facebook/contriever').to(torch.device("cpu"))

import datetime
import json
import arxiv

def summarize_research_direction(personal_info):
    prompt_qa = (
    "Based on the list of the researcher's first person persona from different times, please write a comprehensive first person persona. Focus more on more rescent personas. Be concise and clear (around 300 words)."
    "Here are the personas from different times: {peronalinfo}"
    )

    openai.api_key = KEY
    input = {}
    input['peronalinfo'] = personal_info
    prompt = prompt_qa.format_map(input)
    try:
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6,seed = 42, top_p=0)
    except:
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6,seed = 42, top_p=0)
    content = completion.choices[0].message["content"]
    return content

def get_authors(authors, first_author = False):
    output = str()
    if first_author == False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = authors[0]
    return output
def sort_papers(papers):
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output    

def get_daily_papers(topic,query="slam", max_results=300):
    """
    @param topic: str
    @param query: str
    @return paper_with_code: dict
    """

    # output 
    content = dict() 
    Info = dict() 
    search_engine = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    newest_day = None
    # cnt = 0
    for result in search_engine.results():

        # paper_id       = result.get_short_id()
        paper_title    = result.title
        paper_url      = result.entry_id
        # paper_abstract = result.summary
        
        paper_abstract = result.summary.replace("\n"," ")


        publish_time = result.published.date()
        # if newest_day is not None and not(newest_day == publish_time):
        #     break
        # elif newest_day is None:
        #     newest_day = publish_time
        

        if publish_time in content:
            content[publish_time]['abstract'].append(paper_title+ ": "+paper_abstract)
            content[publish_time]['info'].append(paper_title+": "+paper_url)
            # Info[publish_time].append(paper_title+": "+paper_url)
        else:
            content[publish_time] = {}
            content[publish_time]['abstract'] = [paper_title+ ": "+paper_abstract]
            content[publish_time]['info'] = [paper_title+": "+paper_url]
        # cnt = cnt + 1
            # content[publish_time] = [paper_abstract]
            # Info[publish_time] = 
        # print(publish_time)
        # content[paper_key] = f"|**{publish_time}**|**{paper_title}**|{paper_first_author} et.al.|[{paper_id}]({paper_url})|\n"
    data = content
    # print(cnt)
    
    return data, newest_day
def papertitleAndLink(dataset):
    formatted_papers = []
    i = 0
    # import pdb
    # pdb.set_trace()
    for title in dataset:
        
            # import pdb
            # pdb.set_trace()
        i = i +1
        formatted_papers.append("[%d] "%i + title) 
    # i = 0
    # formatted_papers = [f"{"[%d]"%i + papers}" i = i + 1 for k in dataset.keys() for papers in dataset[k]['info']]
    return ';\n'.join(formatted_papers)

def paperinfo(dataset):
    # for k in dataset.keys():
    formatted_papers = [f"{paper}" for k in dataset.keys() for paper in dataset[k]['abstract']]
    return '; '.join(formatted_papers)

def generate_ideas (trend):
    # prompt_qa = (
    #    "Now you are a researcher with this background {profile}, and here is a high-level summarized trend of a research field {trend}."
    #    "How do you view this field? Do you have any novel ideas or insights?"
    # )

    prompt_qa = (
       "Here is a high-level summarized trend of a research field: {trend}."
       "How do you view this field? Do you have any novel ideas or insights?"
       "Please give me 3 to 5 novel ideas and insights in bullet points. Each bullet points should be concise, containing 2 or 3 sentences."
    )

    openai.api_key = KEY
    content_l = []
    input = {}
    # input['profile'] = profile
    input['trend'] = trend
    prompt = prompt_qa.format_map(input)
    try:
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6,seed = 42, top_p=0)
    except:
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6,seed = 42, top_p=0)
    content = completion.choices[0].message["content"]
    content_l.append(content)
    return content_l

def summarize_research_field(profile, keywords, dataset,data_embedding):
    # papers = paperinfo(dataset)
    query_input = {}
    input = {}
    if profile is None: 
        prompt_qa = (
        "Given some recent paper titles and abstracts. Could you summarize no more than 10 top keywords of high level research backgounds and trends."
        # "Here are the keywords: {keywords}"
        "Here are the retrieved paper abstracts: {papers}"
        )
        query_format = (
        "Given the keywords, retrieve some recent paper titles and abstracts can represent research trends in this field."
        "Here are the keywords: {keywords}"
        )
        # input['keywords'] = keywords
        query_input['keywords'] = keywords
    else:
        prompt_qa = (
        "Given some recent paper titles and abstracts. Could you summarize no more than 10 top keywords of high level research backgounds and trends."
        # "Here is my profile: {profile}"
        # "Here are the keywords: {keywords}"
        "Here are the retrieved paper abstracts: {papers}"
        )
        query_format = (
        "Given the profile of me, retrieve some recent paper titles and abstracts can represent research trends related to my profile."
        "Here is my profile: {profile}"
        # "Here are the keywords: {keywords}"
        )
        query_input['profile'] = profile
        # import pdb
        # pdb.set_trace()
    openai.api_key = KEY
    content_l = []
    
    


    query = query_format.format_map(query_input)

    query_embedding=get_bert_embedding([query])
    # text_chunk_l = dataset
    text_chunk_l = []
    data_embedding_l=[]

    # with open(dataset_path, 'r', encoding='utf-8') as file:
    #     dataset = json.load(file)
    title_chunk = []
    for k in dataset.keys():
        # import pdb
        # pdb.set_trace()
        title_chunk.extend(dataset[k]['info'])
        text_chunk_l.extend(dataset[k]['abstract'])
        data_embedding_l.extend(data_embedding[k])
        # import pdb
        # pdb.set_trace()
        # print(dataset[k]['info'])

    # [p if 'graph' in p else "" for p in dataset[k]['info']]
    chunks_embedding_text_all = data_embedding_l
    ch_text_chunk=copy.copy(text_chunk_l)
    ch_text_chunk_embed=copy.copy(chunks_embedding_text_all)
    num_chunk = 10
    # print("raw_chunk_length: ", raw_chunk_length)

    neib_all = neiborhood_search(ch_text_chunk_embed, query_embedding, num_chunk)

    neib_all=neib_all.reshape(-1)

    context = []
    retrieve_paper = []

    for i in neib_all:
        context.append(ch_text_chunk[i])
        # if i not in retrieve_paper:
        retrieve_paper.append(title_chunk[i])
    # import pdb
    # pdb.set_trace()
    input['papers'] = '; '.join(context)
    prompt = prompt_qa.format_map(input)
    # import pdb
    # pdb.set_trace()
    # import pdb
    # pdb.set_trace()
    
    
    try:
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}],   max_tokens=512)
    except:
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}],   max_tokens= 512)
    content = completion.choices[0].message["content"]
    content_l.append(content)
    return content_l, retrieve_paper
def update_json_file(filename,data_all):
    with open(filename,"r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)
                
    json_data = m.copy() 
    
    # update papers in each keywords         
    for data in data_all:
        for time in data.keys():
            papers = data[time]
            # print(papers.published)i
            cur_time = time.strftime("%m/%d/%Y")
            if cur_time in json_data:
                json_data[time.strftime("%m/%d/%Y")].extend(papers)    
            else:
                json_data[time.strftime("%m/%d/%Y")] = papers
    for time in json_data.keys():
        papers = json_data[time]
        papers['ch_abs']=copy.deepcopy(papers['abstract'])
        # print(papers.published)
        # json_data[time] = papers
    
    with open(filename,"w") as f_:
        json.dump(json_data,f_)
    return json_data

def update_pickle_file(filename, data_all):

    # if os.path.exists(filename):
        # with open(filename,"rb") as f:
        #     m = pickle.loads(f)
    # with open(filename,"rb") as f:
    #     content = f.read()
    #     if not content:
    #         m = {}
    #     else:
    #         m = json.load(content)

    # if os.path.exists(filename):
    with open(filename,"rb") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = pickle.loads(content)
    # else:
    #     with open(filename, mode='w', encoding='utf-8') as ff:
    #         m = {}
    # if os.path.exists(filename):
    #     with open(filename, "rb") as file:
    #         m = pickle.load(file)
    # else:
    #     m = {}

    # json_data = m.copy() 
    # else:
    #     with open(filename, mode='wb', encoding='utf-8') as ff:
    #         m = {}

    # with open(filename, "rb") as file:
    #     m = pickle.load(file)
    pickle_data = m.copy()

    for time in data_all.keys():
        embeddings = data_all[time]
        if time in pickle_data:
            pickle_data[time].extend(embeddings)
        else:
            pickle_data[time] =embeddings
    with open(filename, "wb") as f:
        pickle.dump(pickle_data, f)

    return pickle_data
def json_to_md(filename):
    """
    @param filename: str
    @return None
    """
    
    DateNow = datetime.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-','.')
    
    with open(filename,"r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    md_filename = "README.md"  
      
    # clean README.md if daily already exist else create it
    with open(md_filename,"w+") as f:
        pass

    # write data into README.md
    with open(md_filename,"a+") as f:
  
        f.write("## Updated on " + DateNow + "\n\n")
        
        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            # the head of each part
            f.write(f"## {keyword}\n\n")
            f.write("|Publish Date|Title|Authors|PDF|\n" + "|---|---|---|---|\n")
            # sort papers by date
            day_content = sort_papers(day_content)
        
            for _,v in day_content.items():
                if v is not None:
                    f.write(v)

            f.write(f"\n")
    print("finished")   



def neiborhood_search(corpus_data, query_data, num=8):
    d = 768  # dimension
    neiborhood_num = num
    xq = torch.cat(query_data, 0).cpu().numpy()
    xb = torch.cat(corpus_data, 0).cpu().numpy()
    index = faiss.IndexFlatIP(d)
    xq = xq.astype('float32')
    xb = xb.astype('float32')
    faiss.normalize_L2(xq)
    faiss.normalize_L2(xb)
    index.add(xb)  # add vectors to the index
    D, I = index.search(xq, neiborhood_num)

    return I




def get_passage_conclusion_through_LLM(text, question):
    # prompt_qa = ("Given text:{context},given question:{question},based on this text and question, summarize the above text into a passage so that it can best answer this question.")
    prompt_qa = (
        "Given text:{context},based on this text, summarize the above text into a passage that cannot change its original meaning.")
    openai.api_key = KEY

    input = {}
    input['context'] = text
    input['question'] = question
    prompt = prompt_qa.format_map(input)
    try:
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6, seed = 42)
    except:
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6, seed =42)
    content = completion.choices[0].message["content"]
    # print(content)
    return content


def retain_useful_info(text, question):
    prompt_qa = (
        "Given text:{context},given question:{question},based on this text and question, summarize the text into a sentence  that is most useful in answering this question.")
    openai.api_key = KEY

    input = {}
    input['context'] = text
    input['question'] = question
    prompt = prompt_qa.format_map(input)
    try:
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}])
    except:
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}])
    content = completion.choices[0].message["content"]
    # print(content)
    return content


def llm_summary(text_l):
    # prompt_qa = ("Given text:{context},given question:{question},based on this text and question, summarize the above text into a passage so that it can best answer this question.")
    text = ''
    for inter in text_l:
        text += inter
    prompt_qa = (
        "Given text:{context},based on this text, summarize the above text into a fluent passage that cannot change its original meaning.")
    openai.api_key = KEY

    input = {}
    input['context'] = text
    prompt = prompt_qa.format_map(input)
    try:
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6, seed =42)
    except:
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6, seed=42)
    content = completion.choices[0].message["content"]
    # print(content)
    return content


def get_multi_query_through_LLM(question_data, generated_answers=None, support_material=None):
    PROMPT_DICT = {
        "without_answer": (
            "The input will be a paragraph of text."
            "Your task is to generate five as diverse, informative, and relevant, as possible versions of supporting materials, perspectives, fact. Provide these alternative materials, perspectives, fact. Each of them occupies a line."
            "Original text: {question}"
            "Answer:,Please output a list to split these five answers."),
        "with_answer": (
            "The input will be a paragraph of original text, a previously generated support material and a response for the text based on reviously generated support material  by a naive agent, who may make mistakes."
            "Your task is to generate five as diverse, informative, and relevant, as possible versions of supporting materials,perspectives, fact based on the the above information. Each of them occupies a line."
            "Provide these alternative materials, perspectives, fact."
            "Original text:{question}. "
            "Previously generated support material (the text below are naive, and could be wrong, use with caution): {support_material} "
            "Response:{answer}."
            "Answer:,Please output a list to split these five answers."),
    }
    prompt_q, prompt_qa = PROMPT_DICT["without_answer"], PROMPT_DICT["with_answer"]
    openai.api_key = KEY
    ### question_data
    inter = {}
    inter['question'] = question_data
    if generated_answers != None:
        inter['answer'] = generated_answers
        inter['support_material'] = support_material
        prompt = [prompt_qa.format_map(example) for example in [inter]]
    else:
        prompt = [prompt_q.format_map(example) for example in [inter]]
    try:
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt[0]}], temperature=0.6, seed=42)
    except:
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt[0]}], temperature=0.6,seed =42)
    content = completion.choices[0].message["content"]
    for inter_ in content:
        inter_ = inter_.strip('1.').strip('2.').strip('3.').strip('4.').strip('5.')
    # print(content)

    return content


def get_question_through_LLM(question, context):
    prompt_s = question[0]
    for i in range(len(context)):
        prompt_s += "Documents %d: " % (i + 1) + context[i] + '\n'

    prompt_qa = (prompt_s)

    openai.api_key = KEY
    content_l = []
    # import pdb
    # pdb.set_trace()
    # for inter1 in range(len(context)):

    # question_i = question[0]
    # context_i=context[inter1]
    # input={}
    # input['question']=question_i
    # input['context']=context_i
    prompt = prompt_qa
    try:
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6, seed=42)
    except:
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": prompt}], temperature=0.6, seed=42)
    content = completion.choices[0].message["content"]
    content_l.append(content)
    # print(content)
    return content_l


def get_response_through_LLM(question, context):
    prompt_qa = ("Given text: {context}, based on this text, answer the question: {question}")
    openai.api_key = KEY
    content_l = []
    # print(len(context))
    # import pdb
    # pdb.set_trace()
    # print()

    for inter1 in range(len(question)):
        question_i = question[inter1]
        context_i = context[inter1]
        input = {}
        input['question'] = question_i
        input['context'] = context_i
        prompt = prompt_qa.format_map(input)
        # print(prompt)
        try:
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}], temperature=0.6,seed=42)
        except:
            time.sleep(20)
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}], temperature=0.6,seed=42)
        content = completion.choices[0].message["content"]
        content_l.append(content)
        # print("Answer for Pre Queston ", inter1, ": ")
        # print(content,"\n")
    return content_l

def get_response_through_LLM_answer(question, context, profile):
    # import pdb
    # pdb.set_trace()
    if profile is None:
        prompt_qa = (
            "Answer the: {question}, based on materials: {context}"
        )
    else:
        prompt_qa = (
            "Answer the: {question}, based on materials: {context} and my profile: {profile}"
        )
    openai.api_key = KEY
    content_l = []
    # print(len(context))
    # import pdb
    # pdb.set_trace()
    # print()
    
    # print("Length of the question: ", len(question))
    # print("Length of the context: ", len(context))

    for inter1 in range(len(question)):

        question_i = question[inter1]
        context_i = context[inter1]
        
            
        input = {}
        input['question'] = question_i
        input['context'] = context_i
        if profile is not None:
            profile_i = profile
            input['profile'] = profile_i
            # import pdb
            # pdb.set_trace()
        prompt = prompt_qa.format_map(input)
        # print(prompt)
        try:
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}], temperature=0.6,seed=42)
        except:
            time.sleep(20)
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}], temperature=0.6,seed=42)
        content = completion.choices[0].message["content"]
        content_l.append(content)
        # print(content)
    return content_l

def get_response_through_LLM_cross(question, context):

    prompt_s = context + '\n'
 
    prompt_s += "Based on the above documents, answer the question: {question} in short."
    prompt_qa = (prompt_s)

    openai.api_key = KEY
    content_l = []
    for inter1 in range(len(question)):

        question_i = question[inter1]
        input = {}
        input['question'] = question_i
        prompt = prompt_qa.format_map(input)
        try:
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}], temperature=0.6,seed=42)
        except:
            time.sleep(20)
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}], temperature=0.6,seed=42)
        content = completion.choices[0].message["content"]
        content_l.append(content)
        # print(content)
    return content_l


def get_bert_embedding(instructions):


    # encoded_input_all = [tokenizer(text['instruction']+text['input'], return_tensors='pt').to(torch.device("cuda")) for text in instructions]

    encoded_input_all = [tokenizer(text, return_tensors='pt', truncation=True,
                                   max_length=512).to(torch.device("cpu")) for text in instructions]

    with torch.no_grad():
        emb_list = []
        for inter in encoded_input_all:
            emb = model(**inter)
            emb_list.append(emb['last_hidden_state'].mean(1))
    return emb_list

def calculate_similarity(tensor_list, input_tensor):
    flattened_list = [t.flatten() for t in tensor_list]
    flattened_tensor = input_tensor.flatten()
    cosine_similarities = [F.cosine_similarity(flattened_tensor.unsqueeze(0), t.unsqueeze(0)) for t in flattened_list]

    return cosine_similarities

def response_verify(question, context, verify = False):
    if verify:
        prompt_qa = (
            "Input: Given question:{question}, given answer:{context}. Based on the provided question and its corresponding answer, perform the following steps:"
            "Step 1: Determine if the answer is an actual answer or if it merely indicates that the question cannot be answered due to insufficient information. If the latter is true, just output 'idk' without any extra words "
            "Step 2: If it is a valid answer, succinctly summarize both the question and answer into a coherent knowledge point, forming a fluent passage."
        )
    else:
        prompt_qa = (
            "Given question:{question},given answer:{context},based on the given question and corresponding answer, "
            "summarize them into a knowledge point like a fluent passage.")

    openai.api_key = KEY
    content_l = []

    for inter1 in range(len(question)):

        question_i = question[inter1]
        context_i = context[inter1]
        input = {}
        input['question'] = question_i
        input['context'] = context_i
        prompt = prompt_qa.format_map(input)
        # print(prompt)
        try:
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}], temperature=0.6,seed=42)
        except:
            time.sleep(20)
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}], temperature=0.6,seed=42)
        content = completion.choices[0].message["content"]
        content_l.append(content)
        # print(content)
    return content_l


