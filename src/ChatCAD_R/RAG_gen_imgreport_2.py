### This script generates the thinking chain for medical suggestions on medical images' report (generated from RAG_gen_imgreport_1.py)

from util import *
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import argparse
import random
import openai
from retry import retry
import os


system_message = (
        "You are an AI virtual assistant that can do retrieval augmented generation. You will be given a report on a medical image and need to provide medical suggestions based on it.\n"
        "To accomplish this, you will use an RAG tool, ChatCAD-R, to retrieve from medical database and generate answer.\n"
        "You will first question yourself as a USER from a THIRD perspective about what do you need to do to address this task.\n"
        "And for this questioning, write an answer that you think would be appropriate. Your response's format should be:\n"
        "'Question: your question from a third perspective \nAnswer: <thoughts> your thoughts about the question and the answer, and your intended approach including the tool you choose \n"
        "<actions> [{'tool name': API name, 'tool params': empty dictonary}]\n <values> the final results'. \n"
        "When questioning, you should mimic the user be questioning you how to achieve the goal.\n"
        "The API name in this case should be \"ChatCAD-R\" or None. And you need to explicitly include the brackets like <actions> and strictly follow the template format.\n"
        "You will also include the output of the model and how you would use it to answer the question.\n"
        "Feel free to have more variance in expression when asking the question, elaborating your thoughts, and stating the values, so long as they make sense and meet the previous requirements."
    )

user_q_list = [
    "Can you help me analyze the medical report and give me some suggestions?",
    "Could you assist me in analyzing this medical report and provide some recommendations?",
    "Can you review this report and offer me some advice?",
    "Would you be able to help me interpret the medical report and suggest a course of action?",
    "Can you help me understand the medical report and give me your suggestions?",
    "Can you go through this medical report and recommend some next steps?",
    "Could you take a look at this report and advise me on what to do?",
    "Can you assist in examining this report and provide your insights?",
    "Can you help analyze this medical report and share your suggestions?",
    "Would you be able to review this medical report and offer some guidance?",
    "Can you help me with the analysis of this report and suggest some options?",
    "Could you help me go over this medical report and give me some advice?",
    "Would you assist me in understanding this report and provide some recommendations?",
    "Can you take a look at this medical report and offer me some suggestions?",
    "Could you review the medical report and suggest a course of action?",
    "Would you be able to analyze this medical report and give me your insights?",
    "Can you help me interpret the medical report and recommend some steps?",
    "Could you assist in going through this medical report and provide advice?",
    "Can you look over this medical report and share your recommendations?",
    "Would you help me understand this report and suggest what to do next?",
    "Can you review and analyze this report and give me some guidance?",
    "Could you take a look at this medical report and provide your suggestions?",
    "Would you assist me in examining this medical report and offer your advice?",
    "Can you help interpret this medical report and recommend next steps?",
    "Could you help me review the medical report and suggest some options?",
    "Would you go through this medical report and provide your recommendations?",
    "Can you assist in analyzing this report and offer some guidance?",
    "Could you help me understand and analyze this medical report?",
    "Would you take a look at this medical report and suggest a plan of action?",
    "Can you help me review the medical report and give your suggestions?",
    "Could you go through the report and advise on the next steps?",
    "Would you assist in interpreting this medical report and suggest a course of action?",
    "Can you look at this medical report and offer some advice?",
    "Could you help me with this medical report and provide your insights?",
    "Would you be able to analyze and review this report and offer suggestions?",
    "Can you go over this medical report with me and provide some recommendations?",
    "Could you help me examine this medical report and give some advice?",
    "Would you review this report and suggest a plan of action?",
    "Can you help me understand the details of this medical report and suggest next steps?",
    "Could you assist in reviewing this report and offer some recommendations?",
    "Would you be able to interpret this report and provide guidance?",
    "Can you analyze this medical report and share your suggestions?",
    "Could you help me interpret and review this medical report?",
    "Would you go through this medical report and suggest some next steps?",
    "Can you look over the medical report and provide your advice?",
    "Could you assist me in understanding and analyzing this medical report?",
    "Would you help me go through this report and suggest a course of action?",
    "Can you review and provide insights on this medical report?",
    "Could you take a look at and analyze this medical report?",
    "Would you be able to help me with the analysis of this report and give some advice?",
    "Can you go through this medical report and offer your recommendations?"
]
retriev_q_list = [
    "Please retrieve authoratative knowledge from the Merck Manual.",
    "Can you get authoritative information from the Merck Manual?",
    "Please obtain reliable knowledge from the Merck Manual.",
    "Could you gather expert information from the Merck Manual?",
    "Would you retrieve trusted details from the Merck Manual?",
    "Can you source verified information from the Merck Manual?",
    "Can you look up information in the Merck Manual?",
    "Could you find details in the Merck Manual?",
    "Would you search the Merck Manual for information?",
    "Can you access the Merck Manual and provide authoritative knowledge?",
    "Could you consult the Merck Manual for reliable information?",
    "Would you be able to retrieve expert details from the Merck Manual?",
    "Please gather verified knowledge from the Merck Manual.",
    "Can you refer to the Merck Manual for trusted information?",
    "Could you obtain expert advice from the Merck Manual?",
    "Would you search the Merck Manual for accurate details?",
    "Can you look up reliable knowledge in the Merck Manual?",
    "Please find expert information in the Merck Manual.",
    "Could you get verified data from the Merck Manual?",
    "Would you be able to consult the Merck Manual for precise information?",
    "Can you gather authoritative details from the Merck Manual?",
    "Please retrieve reliable information from the Merck Manual.",
    "Could you obtain trusted knowledge from the Merck Manual?",
    "Would you access expert information from the Merck Manual?",
    "Can you look up trusted details in the Merck Manual?",
    "Please find authoritative data in the Merck Manual.",
    "Could you gather reliable details from the Merck Manual?",
    "Would you be able to access the Merck Manual for expert advice?",
    "Can you retrieve precise information from the Merck Manual?",
    "Please consult the Merck Manual for authoritative knowledge.",
    "Could you get accurate details from the Merck Manual?",
    "Would you gather expert knowledge from the Merck Manual?",
    "Can you obtain trusted information from the Merck Manual?",
    "Please look up verified details in the Merck Manual.",
    "Could you access reliable knowledge from the Merck Manual?",
    "Would you find expert details in the Merck Manual?",
    "Can you get precise information from the Merck Manual?",
    "Please retrieve expert knowledge from the Merck Manual.",
    "Could you consult the Merck Manual for accurate details?"
]

assistant_message_template = (
    "Question: <question>\n\n"
    "Answer:\n\n"
    "<thoughts> It would be beneficial to utilize a medial RAG tool to generate the result.\n\n"
    "<actions> [{'tool name': 'ChatCAD-R', 'tool params': {}}]\n\n"
    "<values> <value>\n\n"
)

values_templates = [
    "Happy to assist. Using the medical report you provided, I'll employ a medical RAG model to generate suggestions from authorized sources. Here's my suggestion:",
    "Delighted to help. With your medical report, I will use a medical RAG model to produce medical advice from reputable sources. My suggestion is:",
    "Pleased to support you. Based on the medical report you've given, I'll utilize a medical RAG model to create suggestions from authorized sources. Here is the suggestion:",
    "I'm here to help. With the medical report provided, I will apply a medical RAG model to formulate medical suggestions from credible sources. This is what I recommend:",
    "Glad to be of assistance. Utilizing the medical report you submitted, I'll use a medical RAG model to derive suggestions from authorized sources. My advice is:",
    "Happy to provide assistance. I will leverage the medical report you provided and a medical RAG model to generate authoritative medical suggestions. Here’s my suggestion:",
    "Eager to assist you. With the medical report in hand, I will use a medical RAG model to offer suggestions from verified sources. Here's the advice I suggest:",
    "It's my pleasure to help. Utilizing your medical report, I will deploy a medical RAG model to generate medical suggestions from authorized sources. Here's my recommendation:",
    "Ready to assist. Based on the medical report you provided, I'll employ a medical RAG model to provide suggestions from trusted sources. This is my suggestion:",
    "I'm pleased to assist. With the provided medical report, I will use a medical RAG model to generate suggestions from authorized sources. Here is my recommendation:"
]


@retry(exceptions=openai.error.RateLimitError, tries=3, delay=2, backoff=2)
def call_gpt4o(system_message, user_message_1, assistant_message_1, user_message_2, assistant_message_2, user_message_3, assistant_message_3, user_message_final, temperature=0.95, max_tokens=200, top_p=0.95, top_k=None):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_1},
        {"role": "assistant", "content": assistant_message_1},
        {"role": "user", "content": user_message_2},
        {"role": "assistant", "content": assistant_message_2},
        {"role": "user", "content": user_message_3},
        {"role": "assistant", "content": assistant_message_3},
        {"role": "user", "content": user_message_final}
    ]
    request_params = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    if top_k is not None:
        request_params["top_k"] = top_k
    response = openai.ChatCompletion.create(**request_params)
    return response['choices'][0]['message']['content']


def multiple_replace(text, replacements):
    regex = re.compile("|".join(map(re.escape, replacements.keys())))
    return regex.sub(lambda match: replacements[match.group(0)], text)

def process_entry(d):
    report = d["report"]
    suggestion = d["suggestion"]
    user_message_final = report
    user_message_1 = "Endotracheal tube is seen terminating approximately 2.7 cm above\n the level of the carina.  Enteric tube seen coursing below the level of the\n diaphragm, inferior aspect not included on the image, but side port appears to\n be in the left upper quadrant in expected location of the stomach.  There is\n left base opacity which may represent combination of pleural effusion and\n atelectasis or underlying consolidation due to aspiration or contusion. \n Additional left perihilar opacity is seen which may also relate to aspiration,\n asymmetric pulmonary edema, or infection.  There is slight blunting of the\n right costophrenic angle and a trace pleural effusion may be present.  The\n cardiac and mediastinal silhouettes are grossly stable.  The superior\n mediastinum remains widened although slightly less prominent as compared to\n this prior study and findings are likely accentuated by supine position in the\n AP technique.  However, if there is clinical concern for acute mediastinal\n injury, CT is more sensitive."
    user_message_2 = "Compared to the prior radiograph, lung volumes are lower, particularly in the\n right lower lobe.  A new right lower lobe opacity on both views is at least\n atelectasis.  Superimposed pneumonia is not excluded.  A small right effusion\n is new.  No pneumothorax.  Cardiomediastinal and hilar silhouettes are normal."
    user_message_3 = "Frontal and lateral views of the chest were obtained.  Triple lead\n left-sided pacemaker is again seen with leads similar in position.  There is\n elevation of the left hemidiaphragm and slight blunting of the left\n costophrenic angle which may be due to a small pleural effusion with overlying\n atelectasis.  Calcifications project over the left mid lung.  No right pleural\n effusion is seen.  The right lung is clear."
    question_1 = random.choice(user_q_list) + " " + random.choice(retriev_q_list)
    question_2 = random.choice(user_q_list) + " " + random.choice(retriev_q_list)
    question_3 = random.choice(user_q_list) + " " + random.choice(retriev_q_list)
    assistant_message_1 = multiple_replace(assistant_message_template, {"<question>": question_1,
                                                                        "<value>": random.choice(values_templates),
                                                                        "<report>": user_message_1})
    assistant_message_2 = multiple_replace(assistant_message_template, {"<question>": question_2,
                                                                        "<value>": random.choice(values_templates),
                                                                        "<report>": user_message_2})
    assistant_message_3 = multiple_replace(assistant_message_template, {"<question>": question_3,
                                                                        "<value>": random.choice(values_templates),
                                                                        "<report>": user_message_3})
    thinkings = call_gpt4o(system_message, user_message_1, assistant_message_1, user_message_2, assistant_message_2, user_message_3, assistant_message_3, user_message_final)
    
    return {
        "id": d["id"],
        "study_id": d["study_id"],
        "subject_id": d["subject_id"],
        "User": user_message_final,
        "Topic": d["topics"],
        "Similarity": d["similarity"],
        "Knowledge": d["knowledge"],
        "Suggestion": thinkings + suggestion
    }

def main(args):
    openai.api_key = args.api_key

    # Path to backup the json file while generating
    directory, filename = os.path.split(args.output_json)
    name, extension = os.path.splitext(filename)
    backup_filename = f"{name}_backup{extension}"
    backup_file_path = os.path.join(directory, backup_filename)

    results = []
    if os.path.exists(args.output_json):
        with open(args.output_json, 'r', encoding='utf-8') as file:
            results = json.load(file)
    with open(args.input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    start_index = len(results)
    data_to_process = data[start_index:]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_entry, d) for d in data_to_process]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processed", unit="iter"):
            result = future.result()
            results.append(result)
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            with open(backup_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    executor.shutdown(wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Enter your OpenAI API key here.")
    parser.add_argument("--workers", type=int, default=5, help="Parallel processing worker number. Adjust based on your OpenAI API tier.")
    parser.add_argument("--input_json", type=str, default="./report_suggestion.json", help="Result from RAG_gen_imgreport_1.py")
    parser.add_argument("--output_json", type=str, default="./RAG_gen_imgreport.json")
    args = parser.parse_args()
    main(args)