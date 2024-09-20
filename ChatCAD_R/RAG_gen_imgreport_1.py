### This script generate the report-suggestion pairs for the medical images in mimic-cxr dataset

from util import *
import argparse
from tqdm import tqdm
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def process_entry(report, info, chatbot):
    try:
        message_history = [{"role": "user", "content": report}]
        user_input = ""
        ref_record = concat_history(message_history)
        response, check, query, abnormality_check, [raw_topic, cos_sim], knowledge = handle_text_input(chatbot, user_input, ref_record, check=True)
        if "请谨慎采纳" in response or check == 0:
            return None
        return {"id": info[0], "study_id":info[1], "subject_id":info[2], "report": report,
                    "brief": abnormality_check, "suggestion": response, "abnormality": check,
                     "topics": list(raw_topic), "similarity": list(np.round(np.array(cos_sim).astype(float),4)), 
                     "disease": query, "knowledge": knowledge}
    except:
        return None

def main(args):
    f = open("./annotation.json")
    annotation_dict = json.load(f)["train"]
    reports, infos = [], []
    repeat, underline, cnt = 0, 0, 0
    for ant in annotation_dict:
        cnt += 1
        if "___" in ant["report"]:
            underline += 1
            continue
        if ant["report"] not in reports:
            reports.append(ant["report"])
            infos.append([ant["id"], ant["study_id"], ant["subject_id"]])
            if len(reports) == args.target_length:
                break
        else:
            repeat += 1

    api_key = args.api_key
    chatbot = initialize_chatbot(api_key)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_entry, reports[i], infos[i], chatbot) for i in range(len(reports))]
        with tqdm(total=len(reports), desc="Processing", unit="report") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                    with open(args.output_json, 'w', encoding='utf-8') as file:
                        json.dump(results, file, cls=CustomEncoder, indent=4, ensure_ascii=False)
                    pbar.update(1)

    print(f"total reports: {cnt}, repeated reports: {repeat}, unique reports: {len(reports)}, reports with underline: {underline}")
    executor.shutdown(wait=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Enter your OpenAI API key here.")
    parser.add_argument("--target_length", type=int, default=100, help="The number of suggestions on unique reports")
    parser.add_argument("--workers", type=int, default=5, help="Parallel processing worker number. Adjust based on your OpenAI API tier.")
    parser.add_argument("--output_json", type=str, default="./report_suggestion.json")
    args = parser.parse_args()
    main(args)
