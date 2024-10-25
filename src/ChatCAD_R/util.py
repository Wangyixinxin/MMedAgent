import sys, os
sys.path.append(os.path.dirname(__file__))
from chat_bot_RAG import gpt_bot

def initialize_chatbot(api_key=None):
    if api_key is None:
        api_key = input("Please enter your OPEN-AI API key: ")
    try:
        chatbot_bindings = gpt_bot(engine="gpt-4o", api_key=api_key)
        print("Chatbot initialized successfully")
        return chatbot_bindings
    except Exception as e:
        print("Failed to initialize openai chatbot: ", e)
        return None

def handle_text_input(chatbot_bindings, input_text, message_history, force_generate=False):
    # yes this function is trivial and unnecessary but it may help in debugging
    response, check, query, abnormality_check, [raw_topic, cos_sim], knowledge = chatbot_bindings.chat_report(input_text, str(message_history), force_generate)
    return response, check, query, abnormality_check, [raw_topic, cos_sim], knowledge


def concat_history(message_history:list)->str:
    ret=""
    for event in message_history:
        ret+=f"{event['role']}: {event['content']}\n"
    return ret

def MRG(image, api_key=None, chatbot=None):
    # get image input and generate medical report, using original chatcad
    if not chatbot:
        chatbot = initialize_chatbot(api_key)
    response, modality = chatbot.report_zh(image)
    return response

def RAG(report, api_key=None, chatbot=None):
    if not chatbot:
        chatbot = initialize_chatbot(api_key)
    message_history = [{"role": "user", "content": report}]
    ref_record=concat_history(message_history)
    ans = chatbot.chat("", ref_record)
    if type(ans) == str:    # casual question
        return ans
    response, check, query, abnormality_check, [raw_topic, cos_sim], knowledge = ans
    return response