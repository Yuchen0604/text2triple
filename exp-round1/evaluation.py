import utils
import os

import eval_utils
import utils

# base directory path
base_dir = r"c:/Users/yuche/OneDrive - TUM/phd/KGC exp/pilot-exp/exp-round1"
data_dir = os.path.join(base_dir, "data")

################### for wikidata-tekgen dataset ###########################
wikidata_dir = os.path.join(data_dir, "wikidata_tekgen")
wikidata_new_dir = os.path.join(data_dir, "wikidata_new")
ontology_file_path = os.path.join(wikidata_dir, "ontologies", "1_movie_ontology.json")
ground_truth_file_path = os.path.join(wikidata_dir, "ground_truth/ont_1_movie_ground_truth.jsonl") 
""" 
#################### for dbpedia-webnlg dataset ###########################
dbpedia_dir = os.path.join(data_dir, "dbpedia_webnlg")
dbpedia_new_dir = os.path.join(data_dir, "wikidata_new")
ontology_file_path = os.path.join(dbpedia_dir, "ontologies", "19_film_ontology.json")
ground_truth_file_path = os.path.join(dbpedia_dir, "ground_truth/ont_19_film_ground_truth.jsonl") """


ds_folder = os.path.join(base_dir, "llm_output", "deepseek-v3")
gemini_folder = os.path.join(base_dir, "llm_output", "gemini-1.5-flash-8b")

ground_truth = utils.load_jsonl(ground_truth_file_path)
ground_truth = eval_utils.convert_to_dict(ground_truth)
ontology = utils.load_json(ontology_file_path)

############################### edit param directory
directory = ds_folder
# directory = gemini_folder 
os.makedirs(os.path.join(directory, "eval"), exist_ok=True)

for filename in os.listdir(directory):
    if filename.endswith(".jsonl"):  # Process only JSON files
        file_path = os.path.join(directory, filename)
        print(f"Processing file: {file_path}")
        system_output = utils.load_jsonl(file_path) 
        system_output = eval_utils.convert_to_dict(system_output)
        
        # initialize the local variables for the evaluation metrics for each file
        t_p, t_r, t_f1, t_onto_conf, t_rel_halluc, t_sub_halluc, t_obj_halluc = 0, 0, 0, 0, 0, 0, 0
        eval_metrics_list = []

        eval_output_path = os.path.join(directory, "eval", f"{filename.split('.')[0]}_eval_output.json")
        metric_path = os.path.join(directory, "eval", f"{filename.split('.')[0]}_metrics.json")
        print(f"eval output path: {eval_output_path}")
        print(f"metric path: {metric_path}")

        for sent_id in list(system_output.keys()):
            # get system output 
            system_triples = system_output[sent_id]['triples']


            # collect the ground truth triples
            if sent_id in ground_truth:
                gt_triples = [[tr['sub'], tr['rel'], tr['obj']] for tr in ground_truth[sent_id]['triples']]
                sentence = ground_truth[sent_id]["sent"]


                # collect the set of relations in ground truth triples, spaces are converted to "_" to make them
                # comparable with system triples
                gt_relations = {tr[1].replace(" ", "_") for tr in gt_triples}

                        
                # filter out any triples in system output that does not match with ground truth relations
                # keep only the triples that have relations in the ground truth
                filtered_system_triples = [tr for tr in system_triples if tr[1] in gt_relations]


                # create a normalized string from subject, relation, object of each triple for comparison
                normalized_system_triples = {eval_utils.normalize_triple(tr[0], tr[1], tr[2]) for tr in filtered_system_triples}
                normalized_gt_triples = {eval_utils.normalize_triple(tr[0], tr[1], tr[2]) for tr in gt_triples}

                # compare the system output triples with ground truth triples and calculate precision, recall, f1
                precision, recall, f1 = eval_utils.calculate_precision_recall_f1(normalized_gt_triples, normalized_system_triples)

                # calculate ontology conformance and relation hallucination
                ont_conformance, rel_hallucination = eval_utils.get_ontology_conformance(ontology, system_triples)


                eval_metrics = {
                        "id": sent_id,
                        "precision": f"{precision:.2f}",
                        "recall": f"{recall:.2f}",
                        "f1": f"{f1:.2f}",
                        "onto_conf": f"{ont_conformance:.2f}",
                        "rel_halluc": f"{rel_hallucination:.2f}",
                        "llm_triples": system_triples,
                        "filtered_llm_triples": filtered_system_triples,
                        "gt_triples": gt_triples,
                        "sent": sentence,
                        "normalized_llm_triples": list(normalized_system_triples),  # Convert set to list
                        "normalized_gt_triples": list(normalized_gt_triples)        # Convert set to list
                }
                eval_metrics_list.append(eval_metrics)

                # aggregate precision, recall, f1 for later averaging
                t_p += precision
                t_r += recall
                t_f1 += f1
                t_onto_conf += ont_conformance
                t_rel_halluc += rel_hallucination

        utils.save_json(eval_metrics_list, eval_output_path)

        total_test_cases = len(system_output)
        print(f"Total test cases: {total_test_cases}")
        # average metrics calculate the average of evaluate metrics for all test cases in a given ontology
        average_metrics = {
                                "avg_precision": f"{t_p/total_test_cases:.2f}",
                                "avg_recall": f"{t_r/total_test_cases:.2f}",
                                "avg_f1": f"{t_f1/total_test_cases:.2f}",
                                "avg_onto_conf": f"{t_onto_conf/total_test_cases:.2f}",
                                "avg_rel_halluc": f"{t_rel_halluc / total_test_cases:.2f}",
                                }
        utils.save_json(average_metrics, metric_path)
        print("---------------------------")


