import json
import re

# load json
def load_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)
    
# load jsonl
def load_jsonl(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]
    
def save_jsonl(data: list, jsonl_path: str):
    with open(jsonl_path, "w") as out_file:
        for item in data:
            out_file.write(f"{json.dumps(item)}\n")
    print(f"Saved JSONL data to {jsonl_path}")

def save_json(data: dict, json_path: str):
    with open(json_path, "w") as out_file:
        json.dump(data, out_file, indent=4)
    print(f"Saved JSON data to {json_path}")

# get domain and range of a relation    
def get_domain_range(relation: str, ontology: dict):
    """
    Returns the domain and range of a given relation label from the ontology data.
    
    :param relation: The label of the relation (string)
    :param data: The ontology data (dictionary loaded from JSON)
    :return: A tuple (domain, range) or (None, None) if relation not found
    """
    # Create a mapping of qid to label for concepts
    concept_map = {concept["qid"]: concept["label"] for concept in ontology.get("concepts", [])}
    
    for rel in ontology.get("relations", []):
        if rel.get("label") == relation:
            domain_id = rel.get("domain")
            range_id = rel.get("range")
            return concept_map.get(domain_id, None), concept_map.get(range_id, None)
    
    return None, None  # Return None if relation is not found

# get n ground truth ontology relations
def get_n_ontology_relations(gt_list: list, id, ontology: dict):
    gt_relations = []
    gt_id = id
    ontology = ontology
    for item in gt_list:
        if item['id'] == gt_id:
            triples = item['triples']
            break
    if triples == None:
        return "No triples found for the given id"        

    for triple in triples:
        gt_relations.append(triple['rel'])

    # remove duplicates
    gt_relations = list(set(gt_relations))
    return gt_relations
    
# get ontology prompt
def get_ontology_prompt(gt_rels: list, ontology: dict):

    onto_relations = ""
    for rel in gt_rels:
        domain, range = get_domain_range(rel, ontology)
        rel_string = rel.replace(" ", "_")
        if domain == None:
            domain = ""
        if range == None:
            range = ""
        onto_relations += f"{rel_string}({domain}, {range}), "
    return onto_relations

def get_one_example_prompt(train_sent):
    example_prompt = "\n\nExample Sentence: " + train_sent['sent']
    train_sent['rel_label'] = train_sent['rel_label'].replace(" ", "_")
    example_prompt += "\nExample Output: " + train_sent['rel_label'] + "(" + train_sent['sub_label'] + "," + train_sent['obj_label'] + ")"
    return example_prompt


def get_three_example_prompt(simil_sent_id: list, train: list):
    three_example_prompt = ""
    for id in simil_sent_id:
        train_sent = get_train_sentence(id, train)
        one_example_prompt = get_one_example_prompt(train_sent)
        three_example_prompt += one_example_prompt + "\n"
    return three_example_prompt
        

# get test prompt
def get_test_prompt(test_sentence):
    test_prompt = "\n\nTest Sentence: " + test_sentence
    test_prompt += "\nTest Output: "
    return test_prompt


def get_similar_sentences(test_sentence_id, test_similar):
    for sim in test_similar:
        if sim == test_sentence_id:
            return test_similar[sim]


def get_train_sentence(simil_sent_id, train_sentences):
    for sent in train_sentences:
        if sent['id'] == simil_sent_id:
            return sent
    
# prepare prompt    
def prepare_prompt_zero_shot(test_sentence: str, gt_relations:list, ontology:dict) -> str:


    prompt_fixed = '''Extract relational triplets from the sentence based on the provided ontology relations.
Use only the listed relations and ensure subjects and objects align with their specified restrictions.
Only return the triples in the format relation(subject, object), separated by commas. Do not include explanations, extra text, or comments. \n
'''

    prompt = prompt_fixed
    prompt += 'CONTEXT:\n\n'
    prompt += '\nOntology Relations: '
    prompt += get_ontology_prompt(gt_relations, ontology)
    prompt += get_test_prompt(test_sentence)

    return prompt


def prepare_prompt_one_shot(test_sentence: str, gt_relations:list, ontology:dict, train_sent:dict) -> str:


    prompt_fixed = '''Extract relational triplets from the sentence based on the provided ontology relations and examples.
Use only the listed relations and ensure subjects and objects align with their specified restrictions.
Only return the triples in the format relation(subject, object), separated by commas. Do not include explanations, extra text, or comments. \n
'''

    prompt = prompt_fixed
    prompt += 'CONTEXT:\n\n'
    prompt += '\nOntology Relations: '
    prompt += get_ontology_prompt(gt_relations, ontology)
    prompt += get_one_example_prompt(train_sent)
    prompt += get_test_prompt(test_sentence)

    return prompt


def prepare_prompt_three_shot(test_sentence: str, gt_relations:list, ontology:dict, simil_sent_id: list, train: list) -> str:


    prompt_fixed = '''Extract relational triplets from the sentence based on the provided ontology relations and examples.
Use only the listed relations and ensure subjects and objects align with their specified restrictions.
Only return the triples in the format relation(subject, object), separated by commas. Do not include explanations, extra text, or comments. \n
'''

    prompt = prompt_fixed
    prompt += 'CONTEXT:\n\n'
    prompt += '\nOntology Relations: '
    prompt += get_ontology_prompt(gt_relations, ontology)
    prompt += get_three_example_prompt(simil_sent_id, train)
    prompt += get_test_prompt(test_sentence)

    return prompt



def parse_result(result_str):
    """
    Convert result string from format relation(subject, object) to a list of triples [subject, relation, object].
    If the input does not match the expected format, return an empty list.
    """
    triples = []
    pattern = r'(\w+)\(([^,]+),\s*([^)]+)\)'
    
    matches = re.findall(pattern, result_str)
    
    if not matches:
        return []  # Return empty list if no valid triples are found

    for relation, subject, obj in matches:
        triples.append([subject.strip(), relation.strip(), obj.strip()])
    
    return triples