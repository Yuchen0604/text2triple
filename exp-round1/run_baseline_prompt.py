from openai import OpenAI
from google import genai
import time
import os
import json
import utils

# folder path
base_dir = r"c:/Users/yuche/OneDrive - TUM/phd/KGC exp/pilot-exp/exp-round1"
data_dir = os.path.join(base_dir, "data")

################## for wikidata-tekgen dataset ###########################
wikidata_dir = os.path.join(data_dir, "wikidata_tekgen")
baseline_dir = os.path.join(wikidata_dir, "baselines")
input_dir = os.path.join(baseline_dir, "prompts") 

category_id = "ont_1_movie"
input_file_name = f"{category_id}_prompts.jsonl"
input_file_path = os.path.join(input_dir, input_file_name)

# read json file
input_data = utils.load_jsonl(input_file_path)

""" ################## for dbpedia-webnlg dataset ###########################
dbpedia_dir = os.path.join(data_dir, "dbpedia_webnlg")
baseline_dir = os.path.join(dbpedia_dir, "baselines")
input_dir = os.path.join(baseline_dir, "prompts")

category_id = "ont_19_film"
input_file_name = f"{category_id}_prompts.jsonl"
input_file_path = os.path.join(input_dir, input_file_name)

# read json file
input_data = utils.load_jsonl(input_file_path) """


""" ######################################### run gemini-1.5-flash-8b model


# open the output folder
output_folder_path = os.path.join(base_dir, "llm_output", "gemini-1.5-flash-8b")	
os.makedirs(output_folder_path, exist_ok=True) 


client = genai.Client(api_key="AIzaSyC7Z2nykkmBUnvGcuugmQu0JMP1EbTMLUo")

# File path to save the results
output_file_path = os.path.join(output_folder_path, f"{category_id}_baseline_results.jsonl")

# Open output file in append mode
with open(output_file_path, "a", encoding="utf-8") as f_out:
    for data in input_data:
        prompt = data["prompt"]
        if prompt is None:
            print(f"Prompt not found")
        else:
            # Prompt sent to the Gemini model
            start_time = time.time()
            generation_config = {
                "temperature": 0,
                "max_output_tokens": 1000
            }
            response = client.models.generate_content(
                model="gemini-1.5-flash-8b", 
                contents=prompt,
                config=generation_config
            )
            end_time = time.time()

            result = response.text
            print(result)
            print("\n")

            run_time = end_time - start_time
            usage = str(response.usage_metadata)
            triples = utils.parse_result(result)

            output_item = {
                "id": data["id"],
                "prompt": prompt,
                "result": result,
                "run_time": run_time,
                "usage_metadata": usage,
                "triples": triples
            }

            # Write output item to file immediately
            f_out.write(json.dumps(output_item) + "\n") """



################################### run deepseek-v3 model

# open the output folder
output_folder_path = os.path.join(base_dir, "llm_output", "deepseek-v3")	
os.makedirs(output_folder_path, exist_ok=True) 

client = OpenAI(api_key="sk-581f14e3b92345799ce10c16bc63eae3", base_url="https://api.deepseek.com")

# File path to save the results
output_file_path = os.path.join(output_folder_path, f"{category_id}_baseline_results.jsonl")

# Open the output file in append mode
with open(output_file_path, "a", encoding="utf-8") as f_out:
    for data in input_data:
        prompt = data["prompt"]
        if prompt is None:
            print(f"Prompt not found")
        else:
            # Prompt sent to the LLM
            start_time = time.time()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0,
                max_tokens=1000
            )
            end_time = time.time()

            result = response.choices[0].message.content
            print(result)
            print("\n")

            run_time = end_time - start_time
            usage = str(response.usage)
            triples = utils.parse_result(result)

            output_item = {
                "id": data["id"],
                "prompt": prompt,
                "result": result,
                "run_time": run_time,
                "usage_metadata": usage,
                "triples": triples
            }

            # Write output item immediately
            f_out.write(json.dumps(output_item) + "\n")