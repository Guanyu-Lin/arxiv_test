import os
import pickle
import json
import time
import datetime
from xml.etree import ElementTree
from huggingface_hub import CommitScheduler
from huggingface_hub import HfApi
from pathlib import Path
import requests
from datasets import load_dataset_builder
import warnings
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import *
import thread6
MAX_DAILY_PAPER = 10
DAY_TIME = 60 * 60 * 24
DAY_TIME_MIN = 60 * 24
DATA_REPO_ID = "cmulgy/ArxivCopilot_data"

DATASET_DIR = Path(".")
DATASET_DIR.mkdir(parents=True, exist_ok=True)
from huggingface_hub import hf_hub_download




def feedback_thought(input_ls): # preload
    agent, query, ansA, ansB, feedbackA, feedbackB = input_ls
    filename_thought = agent.thought_path
    filename = agent.feedback_path

    date = agent.today

    json_data = agent.feedback
    json_data_thought = agent.thought

    if date in json_data:
        if query not in json_data[date]:
            json_data[date][query] = {}
    else:
        json_data[date] = {}        
        json_data[date][query] = {}

    if date not in json_data_thought:
        json_data_thought[date] = []        
    
    
    json_data[date][query]["answerA"] = (ansA)
    json_data[date][query]["feedbackA"] = feedbackA
    json_data[date][query]["answerB"] = (ansB)
    json_data[date][query]["feedbackB"] = feedbackB
    with scheduler.lock:
        with open(filename,"w") as f:
            json.dump(json_data,f)

    preferred_ans = ""
    if feedbackA == 1:
        new_knowledge = response_verify([query], [ansA], verify=False)
        preferred_ans = ansA
        # json_data_thought[date].append(query + ansA)
    else:
        new_knowledge = response_verify([query], [ansB], verify=False)
        preferred_ans = ansB
        # json_data_thought[date].append(query + ansB)

    if ('idk' not in new_knowledge[0]):

        new_knowledge_embedding = get_bert_embedding(new_knowledge)
        thought_embedding_all = []
        for k in agent.thought_embedding.keys():
            thought_embedding_all.extend(agent.thought_embedding[k])

        similarity = calculate_similarity(thought_embedding_all, new_knowledge_embedding[0])

        similarity_values = [s.item() for s in similarity]  # Convert each tensor to a scalar
        if all(s < 0.85 for s in similarity_values):
            # self.update_feedback(an, answer_l_org, query)
            tem_thought = query + preferred_ans
            json_data_thought[date].append(tem_thought)
            if date not in agent.thought_embedding:
                agent.thought_embedding = {}
                agent.thought_embedding[date] = [get_bert_embedding([tem_thought])[0]]
            else:
                agent.thought_embedding[date].append(get_bert_embedding([tem_thought])[0])
            with scheduler.lock:
                with open(filename_thought,"w") as f:
                    json.dump(json_data_thought,f)

                with open(agent.thought_embedding_path, "wb") as f:
                    pickle.dump(agent.thought_embedding, f)

    # return "Give feedback successfully!"

def dailyDownload(agent_ls):

    agent = agent_ls[0]
    while True:
        time.sleep(DAY_TIME)
        data_collector = []
        keywords = dict()
        keywords["Machine Learning"] = "Machine Learning"

        for topic,keyword in keywords.items():

            data, agent.newest_day = get_daily_papers(topic, query = keyword, max_results = MAX_DAILY_PAPER)
            data_collector.append(data)

        json_file = agent.dataset_path

        update_file=update_json_file(json_file, data_collector)

        time_chunks_embed={}

        for data in data_collector:
            for date in data.keys():
                papers = data[date]['abstract']
                papers_embedding=get_bert_embedding(papers)
                time_chunks_embed[date.strftime("%m/%d/%Y")] = papers_embedding
        update_paper_file=update_pickle_file(agent.embedding_path,time_chunks_embed, scheduler)
        agent.paper = update_file
        agent.paper_embedding = update_paper_file
        print("Today is " + agent.newest_day.strftime("%m/%d/%Y"))

def dailySave(agent_ls):
    agent = agent_ls[0]


    while True:
        time.sleep(DAY_TIME)
        with scheduler.lock: 
            with open(agent.trend_idea_path, "w") as f_:
                json.dump(agent.trend_idea, f_)
                
            with open(agent.thought_path, "w") as f_:
                json.dump(agent.thought, f_)

            with open(agent.thought_embedding_path, "wb") as f:
                pickle.dump(agent.thought_embedding, f)
                
            with open(agent.profile_path,"w") as f:
                json.dump(agent.profile,f)
            with open(agent.comment_path,"w") as f:
                json.dump(agent.comment,f)

class ArxivAgent:
    def __init__(self):
        
        self.dataset_path = DATASET_DIR / "dataset/paper.json"
        self.thought_path = DATASET_DIR / "dataset/thought.json"
        self.trend_idea_path = DATASET_DIR / "dataset/trend_idea.json"
        self.profile_path = DATASET_DIR / "dataset/profile.json"
        self.comment_path = DATASET_DIR / "dataset/comment.json"

        self.embedding_path = DATASET_DIR / "dataset/paper_embedding.pkl"
        self.thought_embedding_path = DATASET_DIR / "dataset/thought_embedding.pkl"
        
        self.feedback_path = DATASET_DIR / "dataset/feedback.json"
        self.today = datetime.datetime.now().strftime("%m/%d/%Y")

        self.newest_day = ""
        
        self.load_cache()

        self.download()
        # try:
        #     thread6.run_threaded(dailyDownload, [self])
        #     thread6.run_threaded(dailySave, [self])
        # except:
        #     print("Error: unable to start thread")

    def edit_profile(self, profile, author_name):

        self.profile[author_name]=profile

        return "Successfully edit profile!"
    
    def get_profile(self, author_name):
        if author_name == "": return None

        profile = self.get_arxiv_data_by_author(author_name)
        return profile
    def select_date(self, method, profile_input):
   
        today = self.newest_day
        chunk_embedding_date={}

        
        paper_by_date = {}
        if method == "day":
            offset_day = today 
            str_day = offset_day.strftime("%m/%d/%Y")
            if str_day in self.paper:
                paper_by_date[str_day] = self.paper[str_day]
                chunk_embedding_date[str_day]=self.paper_embedding[str_day]

        elif method == "week":
            for i in range(7):
                offset_day = today - datetime.timedelta(days=i)
                str_day = offset_day.strftime("%m/%d/%Y")
                
                if str_day in self.paper:
                    # print(str_day)
                    paper_by_date[str_day] = self.paper[str_day]
                    chunk_embedding_date[str_day] = self.paper_embedding[str_day]
        else:
            # import pdb
            # pdb.set_trace()
            paper_by_date = self.paper
            chunk_embedding_date=self.paper_embedding

        dataset = paper_by_date
        data_chunk_embedding=chunk_embedding_date
        profile = profile_input

        key_update = list(self.paper.keys())[-1]
        isQuery = False
        if profile in self.trend_idea:
            if key_update in self.trend_idea[profile]:
                if method in self.trend_idea[profile][key_update]:
                    trend = self.trend_idea[profile][key_update][method]["trend"]
                    reference = self.trend_idea[profile][key_update][method]["reference"]
                    idea = self.trend_idea[profile][key_update][method]["idea"]
                    isQuery = True 

        # import pdb
        # pdb.set_trace()
        if not(isQuery):
            trend, paper_link = summarize_research_field(profile, "Machine Learning", dataset,data_chunk_embedding) # trend
            reference = papertitleAndLink(paper_link)
            idea = generate_ideas(trend) # idea
            if profile in self.trend_idea:
                if key_update in self.trend_idea[profile]:
                    if not(method in self.trend_idea[profile][key_update]):
                        self.trend_idea[profile][key_update][method] = {}
                else:
                    self.trend_idea[profile][key_update] = {}
                    self.trend_idea[profile][key_update][method] = {}
            else:
                self.trend_idea[profile] = {}
                self.trend_idea[profile][key_update] = {}
                self.trend_idea[profile][key_update][method] = {}

            self.trend_idea[profile][key_update][method]["trend"] = trend
            self.trend_idea[profile][key_update][method]["reference"] = reference 
            self.trend_idea[profile][key_update][method]["idea"] = idea 


        
        if key_update not in self.thought:
            self.thought[key_update] = []
        if key_update not in self.thought_embedding:
            self.thought_embedding[key_update] = []

        self.thought[key_update].append(trend[0])
        self.thought_embedding[key_update].append(get_bert_embedding([trend])[0])
        self.thought[key_update].append(idea[0])
        self.thought_embedding[key_update].append(get_bert_embedding([idea])[0])

        return trend, reference, idea

    def response(self, data, profile_input):

        query = [data]
        profile = profile_input

        query_embedding=get_bert_embedding(query)
        No_paper = [2**i for i in range(10)]
        optimized_cost = []
        initial_cost = []
        for n in No_paper:
            time_start=time.time()
            self.generate_pair_retrieve_text(query_embedding, n)
            time_end=time.time()
            optimized_cost.append(time_end - time_start)

            time_start=time.time()
            self.generate_pair_retrieve_text_initial(query_embedding, n)
            time_end=time.time()
            initial_cost.append(time_end - time_start)
        print(optimized_cost)
        print(initial_cost)
    def generate_pair_retrieve_text(self, query_embedding, n):
        dataset = self.paper
        text_chunk_l = []
        chunks_embedding_text_all = []
        # cnt = 0
        for k in dataset.keys():
            text_chunk_l.extend(dataset[k]['abstract'][:n])
            chunks_embedding_text_all.extend(self.paper_embedding[k][:n])
            # if cnt >= n: break
            # cnt = cnt + 1
        neib_all = neiborhood_search(chunks_embedding_text_all, query_embedding, num=10)

    def generate_pair_retrieve_text_initial(self, query_embedding, n):
        dataset = self.paper
        text_chunk_l = []
        chunks_embedding_text_all = []
        # cnt = 0
        for k in dataset.keys():
            text_chunk_l.extend(dataset[k]['abstract'][:n])
            chunks_embedding_text_all.extend(get_bert_embedding(dataset[k]['abstract'][:n]))
            # if cnt >= n: break
            # cnt = cnt + 1
        neib_all = neiborhood_search(chunks_embedding_text_all, query_embedding, num=10)
        
    def download(self):
        # key_word = "Machine Learning"
        data_collector = []
        keywords = dict()
        keywords["Machine Learning"] = "Machine Learning"
    
        for topic,keyword in keywords.items():
    
            data, self.newest_day = get_daily_papers(topic, query = keyword, max_results = MAX_DAILY_PAPER)
            data_collector.append(data)
            data_collector.append(data)
            data_collector.append(data)
            data_collector.append(data)
            data_collector.append(data)
            data_collector.append(data)

        json_file = self.dataset_path

        
        # try:
        #     hf_hub_download(repo_id=DATA_REPO_ID, filename="dataset/paper.json", local_dir = ".", repo_type="dataset")
        # except:
        with open(json_file,'w')as a:
            print(json_file)

        update_file=update_json_file(json_file, data_collector)

        # try:
        #     hf_hub_download(repo_id=DATA_REPO_ID, filename="dataset/paper_embedding.pkl", local_dir = ".", repo_type="dataset")
        # except:
        with open(self.embedding_path,'wb')as a:
            print(self.embedding_path)
        time_chunks_embed={}

        for data in data_collector:
            for date in data.keys():
                papers = data[date]['abstract']
                papers_embedding=get_bert_embedding(papers)
                time_chunks_embed[date.strftime("%m/%d/%Y")] = papers_embedding
        update_paper_file=update_pickle_file(self.embedding_path,time_chunks_embed)
        self.paper = update_file
        self.paper_embedding = update_paper_file

    

    def load_cache(self):


        filename = self.feedback_path
        # try:
        #     hf_hub_download(repo_id=DATA_REPO_ID, filename="dataset/feedback.json", local_dir = ".", repo_type="dataset")
        #     with open(filename,"rb") as f:
        #         content = f.read()
        #         if not content:
        #             m = {}
        #         else:
        #             m = json.loads(content)
        # except:
        with open(filename, mode='w', encoding='utf-8') as ff:
            m = {}
        self.feedback = m.copy()

        filename = self.trend_idea_path

        # if os.path.exists(filename):
        # try:
        #     hf_hub_download(repo_id=DATA_REPO_ID, filename="dataset/trend_idea.json", local_dir = ".", repo_type="dataset")    
        #     with open(filename,"rb") as f:
        #         content = f.read()
        #         if not content:
        #             m = {}
        #         else:
        #             m = json.loads(content)
        # except:
        with open(filename, mode='w', encoding='utf-8') as ff:
            m = {}
        self.trend_idea = m.copy()


        filename = self.profile_path
        # if os.path.exists(filename):
        # try:
        #     hf_hub_download(repo_id=DATA_REPO_ID, filename="dataset/profile.json", local_dir = ".", repo_type="dataset")    
        #     with open(filename,"rb") as f:
        #         content = f.read()
        #         if not content:
        #             m = {}
        #         else:
        #             m = json.loads(content)
        # except:
        with open(filename, mode='w', encoding='utf-8') as ff:
            m = {}
        self.profile = m.copy()


        filename = self.thought_path
        filename_emb = self.thought_embedding_path
        # if os.path.exists(filename):
        # try:
        #     hf_hub_download(repo_id=DATA_REPO_ID, filename="dataset/thought.json", local_dir = ".", repo_type="dataset")    
        #     with open(filename,"rb") as f:
        #         content = f.read()
        #         if not content:
        #             m = {}
        #         else:
        #             m = json.loads(content)
        # except:
        with open(filename, mode='w', encoding='utf-8') as ff:
            m = {}

        # if os.path.exists(filename_emb):
        # try:
        #     hf_hub_download(repo_id=DATA_REPO_ID, filename="dataset/thought_embedding.pkl", local_dir = ".", repo_type="dataset")    
        #     with open(filename_emb,"rb") as f:
        #         content = f.read()
        #         if not content:
        #             m_emb = {}
        #         else:
        #             m_emb = pickle.loads(content)
        # except:
        with open(filename_emb, mode='w', encoding='utf-8') as ff:
            m_emb = {}

        self.thought = m.copy() 
        self.thought_embedding = m_emb.copy() 


        filename = self.comment_path
        # if os.path.exists(filename):
        # try:
        #     hf_hub_download(repo_id=DATA_REPO_ID, filename="dataset/comment.json", local_dir = ".", repo_type="dataset")    

        #     with open(filename,"r") as f:
        #         content = f.read()
        #         if not content:
        #             m = {}
        #         else:
        #             m = json.loads(content)
        # except:
        with open(filename, mode='w', encoding='utf-8') as ff:
            m = {}
            
                
        self.comment = m.copy() 



    def update_feedback_thought(self, query, ansA, ansB, feedbackA, feedbackB): 
        try:
            thread6.run_threaded(feedback_thought, [self, query, ansA, ansB, feedbackA, feedbackB])
            # thread6.start_new_thread( print_time, ["Thread-2", 4] )
        except:
            print("Error: unable to start thread")

       
    def update_comment(self, comment):
        date = datetime.datetime.now().strftime("%m/%d/%Y")


                
        json_data = self.comment

        if date not in json_data:
            json_data[date] = [comment]
        else: json_data[date].append(comment) 
        # with scheduler.lock:
        #     with open(filename,"w") as f:
        #         json.dump(json_data,f)
        return "Thanks for your comment!"
    



    def get_arxiv_data_by_author(self, author_name):



        if author_name in self.profile: return self.profile[author_name]
           
        author_query = author_name.replace(" ", "+")
        url = f"http://export.arxiv.org/api/query?search_query=au:{author_query}&start=0&max_results=300"  # Adjust max_results if needed

        response = requests.get(url)
        papers_list = []

        if response.status_code == 200:
            root = ElementTree.fromstring(response.content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')

            total_papers = 0
            data_to_save = []

            papers_by_year = {}

            for entry in entries:

                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                published = entry.find('{http://www.w3.org/2005/Atom}published').text.strip()
                abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                authors_elements = entry.findall('{http://www.w3.org/2005/Atom}author')
                authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in authors_elements]
                link = entry.find('{http://www.w3.org/2005/Atom}id').text.strip()  # Get the paper link

                # Check if the specified author is exactly in the authors list
                if author_name in authors:
                    # Remove the specified author from the coauthors list for display
                    coauthors = [author for author in authors if author != author_name]
                    coauthors_str = ", ".join(coauthors)

                    papers_list.append({
                        "date": published,
                        "Title & Abstract": f"{title}; {abstract}",
                        "coauthors": coauthors_str,
                        "link": link  # Add the paper link to the dictionary
                    })
                authors_elements = entry.findall('{http://www.w3.org/2005/Atom}author')
                authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in authors_elements]

                if author_name in authors:
                    # print(author_name)
                    # print(authors)
                    total_papers += 1
                    published_date = entry.find('{http://www.w3.org/2005/Atom}published').text.strip()
                    date_obj = datetime.datetime.strptime(published_date, '%Y-%m-%dT%H:%M:%SZ')

                    year = date_obj.year
                    if year not in papers_by_year:
                        papers_by_year[year] = []
                    papers_by_year[year].append(entry)

            if total_papers > 40:
                for cycle_start in range(min(papers_by_year), max(papers_by_year) + 1, 5):
                    cycle_end = cycle_start + 4
                    for year in range(cycle_start, cycle_end + 1):
                        if year in papers_by_year:
                            selected_papers = papers_by_year[year][:2]
                            for paper in selected_papers:
                                title = paper.find('{http://www.w3.org/2005/Atom}title').text.strip()
                                abstract = paper.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                                authors_elements = paper.findall('{http://www.w3.org/2005/Atom}author')
                                co_authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in authors_elements if author.find('{http://www.w3.org/2005/Atom}name').text != author_name]

                                papers_list.append({
                                    "Author": author_name,
                                    "Title & Abstract": f"{title}; {abstract}",
                                    "Date Period": f"{year}",
                                    "Cycle": f"{cycle_start}-{cycle_end}",
                                    "Co_author": ", ".join(co_authors)
                                })
  



            # Trim the list to the 10 most recent papers
            papers_list = papers_list[:10]

            # Prepare the data dictionary with the author's name as a key
            # import pdb
            # pdb.set_trace()
            personal_info = "; ".join([f"{details['Title & Abstract']}" for details in papers_list])
            info = summarize_research_direction(personal_info)
            self.profile[author_name] = info

            return self.profile[author_name]

        else:
            return None





    
