from google import genai
import time
import os
import json
import utils

client = genai.Client(api_key="AIzaSyC7Z2nykkmBUnvGcuugmQu0JMP1EbTMLUo")

# folder path
base_dir = r"c:/Users/yuche/OneDrive - TUM/phd/KGC exp/pilot-exp/exp-round1"
# open the output folder
output_folder_path = os.path.join(base_dir, "llm_output", "gemini-1.5-flash-8b")	
os.makedirs(output_folder_path, exist_ok=True) 


""" input_file_name = "ont_19_film_n_rels_0_distractor_3_shot_prompts.jsonl"
input_file_path = os.path.join(base_dir, "data", "dbpedia_new", "n_rels_0_distractor", input_file_name)
output_file_path = os.path.join(output_folder_path, f"{input_file_name.split('.')[0]}_results.jsonl") """


input_file_name = "ont_19_film_baseline_rels_1_shot_prompts.jsonl"
input_file_path = os.path.join(base_dir, "data", "dbpedia_new", "baseline_rels", input_file_name)
output_file_path = os.path.join(output_folder_path, f"{input_file_name.split('.')[0]}_results.jsonl") 

# read json file
input_data = utils.load_jsonl(input_file_path)


# Open file in append mode so we can write results one by one
with open(output_file_path, "a", encoding="utf-8") as f_out:
    for data in input_data:
        prompt = data["prompt"]
        if prompt is None:
            print(f"Prompt not found")
        else:
            # Prompt sent to the LLM
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
            print(response.text)
            print("\n")

            result = response.text
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

            # Write each result immediately as a JSON object per line
            f_out.write(json.dumps(output_item) + "\n")



