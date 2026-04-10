import os
from pathlib import Path

from agent.code.config import SIM_PARAMS
from agent.code.llm_client import LLMClient
import logging
from logging.handlers import RotatingFileHandler

def setup_llm_logger(
    log_dir=None,
    log_file="llm_call_deepseek.log",
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5
):
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("llm_logger")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers (common pitfall with repeated initialization)
    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

class AgentAction:
    """
    Action Module (§4.1.4.2)
    Simulates Task1–Task4 and invokes Memory retrieval, writing, and reflection.
    """
    def __init__(self, profile, memory, base_url=None, api_key=None, model=None, log_name="deepseek.log"):
        self.profile = profile
        self.memory = memory
        self.logger = setup_llm_logger(log_file=log_name)
        if api_key == "local":
            # Local deployment mode:
            # - api_key must be the literal string "local"
            # - base_url is treated as the local model directory path
            #   such as the llama_3b model path in config.py
            try:
                from agent.code.local_llm import LlamaLLM
            except ImportError:
                from local_llm import LlamaLLM
            self.llm = LlamaLLM(model_dir=base_url)
        else:
            self.llm = LLMClient(base_url=base_url, api_key=api_key, model=model)

    def _call_llm(self, messages):
        result = self.llm.call(messages)
        self.logger.info("Messages: %s", messages)
        self.logger.info("LLM Output: %s", result)
        self.logger.info("-" * 80)

        return result

    def simulate_step(self, practice, time_step, similarity_fn):
        # 1. Memory Retrieval
        short_mem = self.memory.retrieve_short()
        long_mem = self.memory.retrieve_long()

        # 2. Build prompt: include profile + short-term + long-term memories
        prompt = self._build_prompt(practice, short_mem, long_mem)
        messages = [
            {'role': 'system', 'content': self.profile.build_prompt()},
            {'role': 'user', 'content': prompt}
        ]

        # 3. LLM generates Task1–Task4
        resp = self._call_llm(messages)

        # 4. Parse results
        ans = {}
        for i in range(1, 5):
            tag = f'Task{i}:'
            if tag in resp:
                ans[f'task{i}'] = resp.split(tag)[1].split('\n')[0].strip()

        # 5. Memory writing: write factual record
        record = [
            practice['exer_content'],
            practice['know_name'],
            practice['score'],
            1
        ]
        self.memory.write_factual(record)

        # 6. Memory reinforcement & long-term update
        # sims = similarity_fn(record)
        if SIM_PARAMS['learning_effect'] == 'yes':
            self.memory.reinforce(record)

        # 7. Forgetting
        if SIM_PARAMS['forgetting_effect'] == 'yes':
            self.memory.forget(time_step)

        # 8. Update Short-term
        self.memory.write_factual(record)

        # 9. Memory Reflection
        # 9a. Corrective Reflection
        corrective = self.memory.reflect_corrective(practice, ans)
        # 9b. Summary Reflection
        reflect = self.memory.reflect_summary(corrective)
        
        messages.append({"role": "assistant",
                        "content": resp})
        messages.append({"role": "user",
                        "content": reflect})
        summary = self._call_llm(messages)
        
        self.memory.write_long_summary(summary)
        self.memory.write_know(practice)

        # 10. Return
        return ans, resp, corrective, summary

    def _build_prompt(self, practice, short_mem, long_mem):
        """
        Build a prompt that follows the paper's Task1–Task4 requirements, including:
        1) Profile prompt
        2) Short-term Memory retrieval results
        3) Long-term Memory retrieval results (reinforced facts, knowledge proficiency, learning status)
        4) Exercise content and instructions for the four tasks
        """
        prompt = (
            f"Recommended Exercise:\n"
            f"- Textual Content: {practice['exer_content']}\n"
            f"- Knowledge Concept (true): {practice['know_name']}\n"
        )

        # Task1: Cognition-driven Action prompt
        prompt += (
            "\nTask 1: Based on your Profile and Knowledge Proficiency, "
            "decide whether you want to attempt this exercise. "
            "If too difficult, output 'No'; otherwise, output 'Yes'.\n"
        )

        # Display short-term memory
        if short_mem:
            prompt += "\nYour Short-term Memory (recent facts):\n"
            for idx, r in enumerate(short_mem, 1):
                prompt += (
                    f" Record {idx}: Content='{r[0]}', Concept={r[1]}, Correct={r[2]}\n"
                )
            prompt += "\n"

        # Task2: Concept identification prompt
        prompt += (
            "Task 2: Identify the knowledge concept tested by this exercise. "
            "Choose one from the following options:\n"
        )
        options = [practice['know_name']] + long_mem.get('practiced_knowledge', [])[:2]
        for opt in options:
            prompt += f" - {opt}\n"
        prompt += "Only output the concept name.\n"

        # Reinforced facts and learning status
        if long_mem.get('significant_facts'):
            prompt += "\nYour Long-term Memory (reinforced facts):\n"
            for idx, f in enumerate(long_mem['significant_facts'], 1):
                prompt += f" Record {idx}: Concept={f[1]}, ReinforcedTimes={f[3]}\n"
        if long_mem.get('learning_status'):
            prompt += (
                f"\nYour current Learning Status Summary: "
                f"{long_mem['learning_status'][-1]}\n"
            )

        # Task3: Problem-solving idea and final answer
        prompt += (
            "\nTask 3: Propose a concise problem-solving idea based on your Profile and Memories, "
            "then give a final answer by selecting EXACTLY ONE option from the given choices.\n"
        )
        
        # Task4: Correctness prediction
        prompt += (
            "\nTask 4: Predict whether you will answer correctly ('Yes' or 'No') based on the idea.\n"
        )

        # Output format specification
        prompt += (
            "\nOutput format exactly as:\n"
            "Task1: <Yes/No>\n"
            "Task2: <concept>\n"
            "Task3: <your idea and final answer>\n"
            "Task4: <Yes/No>\n"
        )
        return prompt

    def close(self):
        if hasattr(self.llm, "close"):
            self.llm.close()
        del self.llm
