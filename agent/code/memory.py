import math, random
from agent.code.config import SIM_PARAMS

class Memory:
    """Memory Module: Factual/Short/Long + retrieval/writing/reflection."""
    def __init__(self, KCG, know_name):
        self.factual = []
        self.short = []
        self.long = {
            'significant_facts': [],
            'learning_status': [],
            'knowledge_proficiency': [],
            'practiced_knowledge': []
        }
        self.threshold = SIM_PARAMS['long_term_thresh']
        self.short_size = SIM_PARAMS['short_term_size']
        self.forget_lambda = SIM_PARAMS['forget_lambda']
        self.KCG = KCG
        self.know_name = know_name

    def write_factual(self, record):
        # Append a new observation
        self.factual.append(record)

    def retrieve_short(self):
        # Return the most recent s records
        self.short = self.factual[-self.short_size:]
        return self.short

    def retrieve_long(self):
        return {
            'significant_facts': self.long['significant_facts'],
            'learning_status':  self.long['learning_status'],
            'knowledge_proficiency': self.long['knowledge_proficiency'],
            'practiced_knowledge': self.long['practiced_knowledge']
        }

    def write_long_summary(self, summary):
        self.long['learning_status'].append(summary)
    
    def write_know(self, practice):
        # practice['know_name'] is a list
        know_list = [
            k.replace('"', '').strip().lower()
            for k in practice['know_name']
        ]

        # Avoid duplicate writes
        for k in know_list:
            if k not in self.long['practiced_knowledge']:
                self.long['practiced_knowledge'].append(k)

        
    # def summarize_similarity_llm_kcg(self, record):
    #     # self._store_log('\n\n' + '-' * 40 + '\nStart memory enhancement\n' + '-' * 40 + '\n\n')
    #     # record1 = '# knowledge concept 1 #: ' + record[1] + '\n'
    #     sim = []
    #     for memory_element in self.factual:
    #         # record2 = '# knowledge concept 2 #: ' + memory_element[1] + '\n'
    #         # print ((self.know_name[record[1].replace('\"','').strip().lower()], self.know_name[memory_element[1].replace('\"','').strip().lower()]))
    #         for know_name_single in record[1]:
    #             if (self.know_name[know_name_single.replace('\"','').strip().lower()], self.know_name[memory_element[1].replace('\"','').strip().lower()]) in self.KCG or (self.know_name[memory_element[1].replace('\"','').strip().lower()], self.know_name[know_name_single.replace('\"','').strip().lower()]) in self.KCG:
    #                 sim.append(1)
    #             # self._store_log(record1 + ' and ' + memory_element[1] + ' are similar.\n')
    #         # else:
    #         #     if self.know_course_list[record[1].replace('\"','').strip().lower()] == self.know_course_list[memory_element[1].replace('\"','').strip().lower()]:
    #         #         random_float = random.uniform(0, 1)
    #         #         if random_float > 0.8:
    #         #             sim.append(1)
    #                     # self._store_log(record1 + ' and ' + memory_element[1] + ' are similar.\n')
    #         else:
    #             sim.append(0)
    #             # self._store_log(record1 + ' and ' + memory_element[1] + ' are dissimilar.\n')
    #     return sim
    
    def summarize_similarity_llm_kcg(self, record):
        sim = []
        for memory_element in self.factual:
            is_similar = 0  # Default: not similar
            # Iterate over each concept in record[1] and memory_element[1]
            for know_name_r in record[1]:
                name_r = self.know_name[know_name_r.replace('\"','').strip().lower()]
                for know_name_m in memory_element[1]:
                    name_m = self.know_name[know_name_m.replace('\"','').strip().lower()]
                    # If any pair exists in the KCG, mark as similar
                    if (name_r, name_m) in self.KCG or (name_m, name_r) in self.KCG:
                        is_similar = 1
                        break  # One similar pair is enough (break inner loop)
                if is_similar:
                    break  # Break outer loop as well
            sim.append(is_similar)

        return sim


    def summarize_similarity_llm(self, record):
        # TODO: Call an LLM-based similarity function or implement a custom similarity rule
        # Return a 0/1 list aligned with self.factual
        sim = [0]*len(self.factual)
        return sim
        

    def reinforce(self, record):      
        # 1) Similarity
        sim_list = self.summarize_similarity_llm_kcg(record)
        # 2) Reinforce old facts
        for i, sim in enumerate(sim_list):
            self.factual[i][3] += sim
        # 3) Add new fact
        max_count = max((r[3] for r in self.factual), default=1)
        self.factual.append([record[0], record[1], record[2], max_count])
        # 4) Migrate to long-term memory
        existing = {f[-1] for f in self.long['significant_facts']}
        for idx, rec in enumerate(self.factual):
            if rec[3]>=self.threshold and (idx+1) not in existing:
                rec.append(idx+1)
                self.long['significant_facts'].append(rec)
                rec[3]=1
        return sim_list

    def forget(self, t):
        kept=[]
        for fact in self.long['significant_facts']:
            ts=fact[4]
            if 1/(1+math.exp(-(t-ts)))<=self.forget_lambda:
                kept.append(fact)
            else:
                self.factual[ts][3]=1
        self.long['significant_facts']=kept

    def reflect_corrective(self, practice, ans):
        fb = ''
        
        # know_name is a list; task2 is the concept identified by the model
        predicted_k = (ans.get('task2') or '').strip().lower()
        
        # Normalize concepts in the list to lowercase for comparison
        real_k_list = [k.replace('"','').strip().lower() for k in practice['know_name']]
        
        # Treat as correct if there is any match
        if predicted_k not in real_k_list:
            fb += (
                'The knowledge tested by this question is ' +
                ', '.join(real_k_list) +
                ' but you wrongly think the knowledge is ' +
                (predicted_k if predicted_k else 'unknown knowledge') +
                '. \n'
            )
        
        # Keep original logic
        if (ans.get('task4','').lower() != 'yes') and practice['score'] == 1:
            fb += "You thought you couldn't solve this problem correctly, but in fact, you will solve it correctly. \n"

        if (ans.get('task4','').lower() != 'no') and practice['score'] == 0:
            fb += "You thought you could solve this problem correctly, but in fact, you does not answer it correctly. \n"

        return fb


    def reflect_summary(self, corrective):
        if corrective != '':
            summary = corrective
        else:
            summary = ''
        summary+=f"\n You should directly output your reflection and summarize your # Learning Status # within 500 words based on your # profile #, # short-term memory #, # long-term memory # and previous # Learning Status #. Do not output any other information."

        self.long['learning_status'].append(summary)
        return summary