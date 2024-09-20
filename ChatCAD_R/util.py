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

def handle_text_input(chatbot_bindings, input_text, message_history, check=False):
    # yes this function is trivial and unnecessary but it may help in debugging
    if not check:
        response, [raw_topic, cos_sim], knowledge = chatbot_bindings.chat(input_text, str(message_history))
        return response, [raw_topic, cos_sim], knowledge
    else:
        response, check, query, abnormality_check, [raw_topic, cos_sim], knowledge = chatbot_bindings.chat(input_text, str(message_history), abnormality_check=True)
        return response, check, query, abnormality_check, [raw_topic, cos_sim], knowledge


def concat_history(message_history:list)->str:
    ret=""
    for event in message_history:
        ret+=f"{event['role']}: {event['content']}\n"
    return ret

def MRG(image, api_key=None):
    # get image input and generate medical report, using original chatcad
    chatbot = initialize_chatbot(api_key)
    response, modality = chatbot.report_zh(image)
    return response

def RAG_report(report, api_key=None):
    # RAG with report input
    chatbot = initialize_chatbot(api_key)
    message_history = [{"role": "user", "content": report}]
    ref_record=concat_history(message_history)
    response, check, query, abnormality_check, knowledge = chatbot.chat_report("", ref_record)
    if not check:
        return "No abnormality that requires further concerns was detected. \nHere is some info relavant to your report that might be interesting:\n"+response
    else:
        return response

def RAG_casual(question, api_key=None):
    # RAG with casual input and medical advice output
    chatbot = initialize_chatbot(api_key)
    message_history = [{"role": "user", "content": question}]
    ref_record=concat_history(message_history)
    response, [raw_topic, cos_sim], knowledge = chatbot.chat("", ref_record)
    return response
