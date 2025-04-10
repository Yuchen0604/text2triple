# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import time
import os
import json
import utils

client = OpenAI(api_key="sk-581f14e3b92345799ce10c16bc63eae3", base_url="https://api.deepseek.com")

# folder path
base_dir = r"c:/Users/yuche/OneDrive - TUM/phd/KGC exp/pilot-exp/exp-round1"
input_file_name = "ont_1_movie_baseline_rels_1_shot_prompts.jsonl"
input_file_path = os.path.join(base_dir, "data", "wikidata_new", "baseline_rels", input_file_name)

# read json file
input_data = utils.load_jsonl(input_file_path)

# open the output folder
output_folder_path = os.path.join(base_dir, "llm_output", "deepseek-v3")	
os.makedirs(output_folder_path, exist_ok=True) 


# Open output file in append mode
output_file_path = os.path.join(output_folder_path, "ont_1_movie_baseline_rels_1_shot_results.jsonl")
with open(output_file_path, "a", encoding="utf-8") as f_out:  # use .jsonl for line-by-line records
    for data in input_data:
        prompt = data["prompt"]
        if prompt is None:
            print(f"Prompt not found")
        else:
            # Prompt sent to the LLM
            start_time = time.time()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0,
                max_tokens=1000
            )
            end_time = time.time()
            print(f"run time: {end_time - start_time}\n")

            result = response.choices[0].message.content
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

            # Write to file immediately
            f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")





