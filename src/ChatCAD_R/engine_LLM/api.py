import os
import json
import pickle as pkl
import numpy as np
import time
import openai
from typing import List, Optional


def get_rules():
    '''Add your questing rules here.
    '''
    # feature_list=["症状和体征","诊断","预后","治疗"]
    info_head="概述"
    no="的"
    identity="图书管理员"
    found="{identity}找到了"
    not_found="{identity}需要其他资料"

    query_1='''
    我想让你扮演一位虚拟的{identity}。不同于给出诊断结果的专业医生，{identity}自身没有任何的医学知识，也无法回答患者的提问。因此，请忘记你的医学知识。现在你必须从知识库中，查阅与患者的问题最可能有帮助的医学知识。我将扮演知识库，告诉你医学知识，以及可以查询的主题。你需要在我提供的选项中，选择一个查询主题，我将告诉你新的医学知识，以及新的可以查询的主题。请重复以上流程，直到你认为，你从我查询到的医学知识，对患者的提问可能有帮助，此时，请告诉我'{found}'注意，你是一个{identity}，无法回答患者的提问，你必须从我扮演的知识库提供的选项中，选择一个医学知识的主题进行查询。患者的问题是："{quest}"你需要尽量查询与这个问题有关的知识。

    现在，你必须选择以下一个主题选项中，选择最可能有帮助的主题回复我:  
    {topic_list}
    注意，你不允许回答患者的提问，不允许回复其他内容，不允许提出建议，不允许回复你做出选择的原因，解释或者假设。你只允许从我扮演的知识库提供的主题选项中，选择一项查询。只回复你想查询的主题选项的名字。
    '''

    query_topic1='''
    如果你认为，你已经查询到了，对于"{quest}"这个问题，可能有帮助的医学知识，请回复我'{found}'如果还没有，你必须选择以下一个主题选项中，选择最可能有帮助的主题回复我:  
    {topic_list} '{not_found}'
    只回复你想查询的选项的名字就可以了，不需要回复别的内容。
    '''

    query_topic2='''
    如果你认为，你已经查询到了，对于"{quest}"这个问题，可能有帮助的医学知识，请回复我'{found}'如果还没有，你必须回复我'{not_found}'。只回复你想查询的选项的名字就可以了，不需要回复别的内容。
    '''

    query_2='''
    做得很好！
    你查询到的医学知识是：
    \'''
    {knowledge}
    \'''
    {query_topic}
    '''

    #v2.5
    query_res=''' 
    请告诉我，刚才你查询到的哪些医学知识，对"{quest}"可能有帮助？请打印\'''内的原文，不要打印别的内容。
    '''

     #v3
    # query_res='''
    # 请打印刚才查询到的医学知识，这些知识应当对"{quest}"可能有帮助。不要打印别的内容。
    # '''
    return locals()
global_rules=get_rules()

def format_query(query,verbose=False,**kwargs):
    # format_pool={
    #     key:globals()[key] for key in keylist
    # }
    # format_pool=global_rules
    # print(kwargs)
    kwargs={**global_rules,**kwargs}
    if verbose: print(query)
    while '{' in query:
        query=query.format(**kwargs)
        if verbose: print(query)
    return query
def list2str(word_list:list):
    ret=" ".join(map(lambda x:f"'{x}'",word_list))
    return ret
    

class ProxyEnvironment:
    def __init__(self, http_proxy, https_proxy):
        self.http_proxy = http_proxy
        self.https_proxy = https_proxy
        self.old_http_proxy = None
        self.old_https_proxy = None

    def __enter__(self):
        self.old_http_proxy = os.getenv("HTTP_PROXY")
        self.old_https_proxy = os.getenv("HTTPS_PROXY")
        
        os.environ["HTTP_PROXY"] = self.http_proxy
        os.environ["HTTPS_PROXY"] = self.https_proxy

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_http_proxy is not None:
            os.environ["HTTP_PROXY"] = self.old_http_proxy
        else:
            os.environ.pop("HTTP_PROXY", None)
        
        if self.old_https_proxy is not None:
            os.environ["HTTPS_PROXY"] = self.old_https_proxy
        else:
            os.environ.pop("HTTPS_PROXY", None)

class Chat_api:
    def __init__(self, api_key: str, proxy: Optional[str] = None, verbose: bool = False, model: str = "gpt-4o"):
        """
        Initialize the ChatApi with OpenAI API key, optional proxy, verbosity, and model selection.

        :param api_key: OpenAI API key.
        :param proxy: Optional proxy URL.
        :param verbose: If True, print prompts and responses.
        :param model: The OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo").
        """
        self.api_key = api_key
        self.verbose = verbose
        self.model = model
        self.now_query = ""
        self.now_res = ""
        self.messages = []  # To keep track of the conversation

        openai.api_key = self.api_key

        # Configure proxy if provided
        self.proxy = proxy

    def prompt(self, query: str, **kwargs):
        """
        Format and store the current query.

        :param query: The user's input query.
        :param kwargs: Additional keyword arguments for formatting.
        """
        formatted_query = format_query(query, **kwargs)
        self.now_query = formatted_query
        self.messages.append({"role": "user", "content": self.now_query})
        if self.verbose:
            print("Human:\n", self.now_query, flush=True)

    def get_res(self, max_connection_try: int = 5, fail_sleep_time: int = 10) -> Optional[str]:
        """
        Send the current query to OpenAI and retrieve the response with retry logic.

        :param max_connection_try: Maximum number of retry attempts.
        :param fail_sleep_time: Seconds to wait before retrying after a failure.
        :return: The response from ChatGPT or None if failed.
        """
        attempt = 0
        while attempt < max_connection_try:
            try:
                if self.proxy:
                    with ProxyEnvironment(self.proxy, self.proxy):
                        response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=self.messages
                        )
                else:
                    response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=self.messages
                        )
                reply = response.choices[0].message.content.strip()
                self.now_res = reply
                self.messages.append({"role": "assistant", "content": self.now_res})
                if self.verbose:
                    print("ChatGPT:\n", self.now_res, flush=True)
                    print()
                return self.now_res
            except openai.error.OpenAIError as e:
                attempt += 1
                if self.verbose:
                    print(f"Warning: OpenAI Connection Error (Attempt {attempt}/{max_connection_try}): {e}")
                if attempt < max_connection_try:
                    time.sleep(fail_sleep_time)
                else:
                    print("Error: Failed to get response from OpenAI after multiple attempts.")
                    return None

    def get_choice_res(self, possible_res: List[str], max_false_time: int = 5) -> Optional[str]:
        """
        Provide several choices to ChatGPT and select one based on the response.

        :param possible_res: A list of possible response choices.
        :param max_false_time: Maximum number of attempts to get a valid choice.
        :return: The selected response choice or None if not found.
        """
        possible_res_formatted = [format_query(q) for q in possible_res]

        def check_res(res: str, possible_res_list: List[str]) -> Optional[str]:
            """
            Check if any of the possible responses are present in the response.

            :param res: The response string from ChatGPT.
            :param possible_res_list: List of possible valid responses.
            :return: The matched response or None.
            """
            punctuation = ",，.。'‘’/、\\:：\"“”?？!！;；`·~@#$%^&*()_+-=<>[]{}|"
            translator = str.maketrans(punctuation, ' ' * len(punctuation))
            res_clean = res.translate(translator)
            res_tokens = res_clean.split()

            for option in possible_res_list:
                if option in res_tokens:
                    return option
            return None

        for attempt in range(max_false_time):
            response = self.get_res()
            if response is None:
                continue  # Skip to the next attempt if response failed
            selected_choice = check_res(response, possible_res_formatted)
            if selected_choice:
                if self.verbose:
                    print("Choice of ChatGPT:", selected_choice)
                    print(flush=True)
                return selected_choice
            if len(self.messages) >= 2:
                self.messages.pop()
                if self.verbose:
                    print("Rolled back the last assistant response due to invalid choice.", flush=True)
        if self.verbose:
            print("Warning: ChatGPT didn't provide a valid response from the possible choices.", flush=True)
        return None


def answer_quest(quest: str,api_key: str,topic_base_dict: list):#,topic):
    global_rules['quest']=quest


    feature_list,info_head,no,quest,topic,identity,found,not_found,query_1,query_topic1,query_topic2,query_2,query_res=global_rules.get('feature_list'),global_rules.get('info_head'),global_rules.get('no'),global_rules.get('quest'),global_rules.get('topic'),global_rules.get('identity'),global_rules.get('found'),global_rules.get('not_found'),global_rules.get('query_1'),global_rules.get('query_topic1'),global_rules.get('query_topic2'),global_rules.get('query_2'),global_rules.get('query_res')
    
    infobase=json.load(open(os.path.join(os.path.dirname(__file__), 'dataset', 'disease_info.json'),"r",encoding="utf-8"))

    # Set proxy if needed : 
    # chatapi=Chat_api(api_key=api_key, verbose=False, proxy='socks5h://127.0.0.1:1080')
    chatapi=Chat_api(api_key=api_key, verbose=False, proxy=None)
    prompt=chatapi.prompt
    get_res=chatapi.get_res
    get_choice_res=chatapi.get_choice_res
    info_topic=""
    
    # topic_list=list(topic_base_dict.keys())
    topic_list=topic_base_dict
    infobase={i:infobase[i] for i in topic_base_dict}

    prompt(query_1,topic_list=list2str(topic_list))
    now_res=get_choice_res([found,not_found]+topic_list)
    if now_res in topic_base_dict:
        info_topic=now_res

    info_list=[infobase]
    while len(info_list)!=0:
        
        now_info=info_list[-1]

        if now_res==format_query(found):
            prompt(query_res)
            found_data=get_res()
            # print(found_data)
            # return now_info_str,found_data
            return info_topic,found_data
            # break
        elif now_res==format_query(not_found):
            info_list.pop()
            if len(info_list)==0:
                # print("not found")
                break
            now_info=info_list[-1]
            topic_list=list(now_info.keys())
            if info_head in topic_list:topic_list.remove(info_head)
            prompt(query_topic1,topic_list=list2str(topic_list))
            possible_res=[found,not_found]+topic_list

        elif now_res in topic_list:
            # now_info=now_info[now_res]
            if type(now_info[now_res])==str:
                now_info_str=now_info.pop(now_res)
                now_info={info_head:now_info_str}
                info_list.append(now_info)
                topic_list=[]
                prompt(query_2,knowledge=now_info_str,query_topic=query_topic2)
                possible_res=[found,not_found]
            else:
                now_info=now_info.pop(now_res)
                topic_list=list(now_info.keys())
                info_list.append(now_info)
                if info_head in topic_list:
                    topic_list.remove(info_head)
                    now_info_str=now_info[info_head]
                    if len(topic_list)==0:
                        prompt(query_2,knowledge=now_info_str,query_topic=query_topic2)
                        # possible_res=[found,not_found]
                    else:
                        prompt(query_2,knowledge=now_info_str,query_topic=query_topic1,topic_list=list2str(topic_list))
                else:
                    prompt(query_topic1,topic_list=list2str(topic_list))
                possible_res=[found,not_found]+topic_list
            
        else:
            # print("unhandle strange result")
            break

        now_res=get_choice_res(possible_res)
        if now_res in topic_base_dict:
            info_topic=now_res
            # topic_list=now_info[now_res].keys()
            # prompt(query_2,knowledge=now_info[now_res],query_topic=query_topic2)
    
    return None


def query_range(model, query: str,k:int=3,bar=0.6):
    msd=json.load(open(os.path.join(os.path.dirname(__file__), 'dataset', 'disease_info.json'),"r",encoding='utf-8'))
    emb_d = pkl.load(open(os.path.join(os.path.dirname(__file__), 'dataset', 'MSD.pkl'),'rb'))
    embeddings=[]
    for key,value in emb_d.items():
        embeddings.append(value)
    embeddings=np.asarray(embeddings)
    # m = SentenceModel()
    q_emb = model.encode(query)
    # q_emb = m.encode(query)
    q_emb=q_emb/np.linalg.norm(q_emb, ord=2)

    # Calculate the cosine similarity between the query embedding and all other embeddings
    cos_similarities = np.dot(embeddings, q_emb) 

    # Get the indices of the embeddings with the highest cosine similarity scores
    top_k_indices = cos_similarities.argsort()[-k:][::-1]
    # print(f"cos similarities of top k choices; only > {bar} will be selected :")
    # print(cos_similarities[top_k_indices])
    sift_topK=top_k_indices[np.argwhere(cos_similarities[top_k_indices]>bar)]
    sift_topK=sift_topK.reshape(sift_topK.shape[0],)
    ret, raw_ret = [], []
    if len(sift_topK)==0:
        return ret, [None,None]
    for indices in sift_topK:
        key=list(emb_d.keys())[indices]
        ret.append(key)
    for indices in top_k_indices:
        key=list(emb_d.keys())[indices]
        raw_ret.append(key)
    return ret, [raw_ret, cos_similarities[top_k_indices]]
