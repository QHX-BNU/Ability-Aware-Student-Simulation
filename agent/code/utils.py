import json, os

def load_json(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Example similarity function for the Memory Module in the paper
def compute_similarity(record, memory_list, kcg, know_course_list):
    '''
    Compute concept similarity between `record` and each fact in `memory_list`.
    kcg: knowledge concept graph (a set of tuple pairs)
    know_course_list: mapping from concept -> course
    record: [text, concept, correct_flag, counter]
    memory_list: [[text, concept, correct_flag, counter], ...]
    Returns: a [0/1 list]
    '''
    sim = []
    rec = record[1].lower().strip().replace('\"','')
    rec_id = know_course_list.get(rec)
    for m in memory_list:
        mem = m[1].lower().strip().replace('\"','')
        mem_id = know_course_list.get(mem)
        if (rec_id, mem_id) in kcg or (mem_id, rec_id) in kcg:
            sim.append(1)
        else:
            # Within the same course, treat as similar with 20% probability
            if rec_id == mem_id and random.random() > 0.8:
                sim.append(1)
            else:
                sim.append(0)
    return sim

# Optional: LLM-based similarity (for more accurate text similarity, call _response_llm_gpt)
# def llm_compute_similarity(record, memory_list, llm_client):
#     pass