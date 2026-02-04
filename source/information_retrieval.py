import os
import sys
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import json
from tools.icrag import initiate_RAG
from tools.reasoning_retrieval import initiate_RRAG
import pandas as pd
from tools.escalation_llm import Escalation_LLM
from tools.single_llm import Single_LLM
from tools.gpt_inference import do_inference

def load_all_files():
    with open('PATH_TO_NFR') as jsonfile:
        nfr = json.load(jsonfile)
    with open('PATH_TO_PERSONA') as jsonfile:
        user_persona = json.load(jsonfile)
    with open('PATH_TO_USER_STORY') as jsonfile:
        user_story = json.load(jsonfile)
    with open('PATH_TO_DOMAIN_DOCS') as jsonfile:
        domain_knowledge = json.load(jsonfile)
    with open('PATH_TO_BACKGROUND') as jsonfile:
        back_ground = json.load(jsonfile)

    return user_story, nfr, user_persona, domain_knowledge, back_ground

def do_reasoning(user_query, docs, reasoning_type):
    enhanced_docs = []
    for doc in docs:
        prompt = f"""You are a software assistant analyzing whether retrieved {reasoning_type} helps address a user's query based on user's situation.
                
                    **User Query:**
                    {user_query[0]}
                    
                    **User Query:**
                    {user_query[1]}
    
                    **Retrieved Documentation:**
                    {doc}
    
                    **Task:**
                    Analyze through three processes and output valid JSON:
    
                    1. **Relevance-Aware Process**: Determine if the {reasoning_type} is relevant to the query and explain why.
                    2. **Evidence-Aware Selective Process**: If relevant, extract key snippets from the {reasoning_type} and explain how each helps answer the query.
                    3. **Trajectory Analysis Process**: Synthesize the information and provide a final answer.
    
                    **Output Format (JSON):**
                    {{
                      "relevance_aware": {{
                        "is_relevant": boolean,
                        "reasoning": "Explanation of why the {reasoning_type} is or isn't relevant to the query"
                      }},
                      "evidence_aware": [
                        {{
                          "cited_snippet": "Exact quote from {reasoning_type}",
                          "how_it_helps": "Detailed explanation of how this snippet addresses the query"
                        }}
                      ],
                      "trajectory_analysis": {{
                        "synthesis": "Comprehensive analysis synthesizing all cited evidence",
                        "final_answer": "Direct answer to the user's query based on the {reasoning_type}"
                      }}
                    }}
    
                    **Important Notes:**
                    - If not relevant, set "is_relevant" to false and return empty array for "evidence_aware"
                    - Extract 1-3 most relevant snippets only
                    - Keep snippets concise but complete enough to be meaningful
                    - Ensure all JSON is valid and parseable
    
                    **Your Response (JSON only):**"""

        enhanced_docs.append({
            'text': user_query,
            'explanation': do_inference(prompt)
        })
    return enhanced_docs

def do_retrival(user_queries, user_situations, n):
    total_results = []

    user_story, nfr, user_persona, domain_knowledge, back_ground = load_all_files()
    vanilla_rag = initiate_RAG()

    for query_num, (user_intention, user_situation) in enumerate(zip(user_queries, user_situations)):
        relevant_persona = vanilla_rag.query([user_intention, user_situation], user_persona)
        relevant_user_story = vanilla_rag.query([user_intention, user_situation], user_story)
        relevant_domain_knowledge = vanilla_rag.query([user_intention, user_situation], domain_knowledge)
        relevant_back_ground = vanilla_rag.query([user_intention, user_situation], back_ground)
        relevant_nfr = vanilla_rag.query([user_intention, user_situation], nfr)

        enhanced_persona = do_reasoning([user_intention, user_situation], relevant_persona, 'User Persona')
        enhanced_user_story = do_reasoning([user_intention, user_situation], relevant_user_story, 'User Story')
        enhanced_domain_knowledge = do_reasoning([user_intention, user_situation], relevant_domain_knowledge, 'Domain Knowledge')
        enhanced_back_ground = do_reasoning([user_intention, user_situation], relevant_back_ground, 'Background')
        enhanced_nfr = do_reasoning([user_intention, user_situation], relevant_nfr, 'Non-functional Requirements')

        enhanced_results = {
            'user_situation': user_situation,
            'user_query': user_intention,
            'persona': enhanced_persona,
            'user_story': enhanced_user_story,
            'domain_knowledge': enhanced_domain_knowledge,
            'background': enhanced_back_ground,
            'nfr': enhanced_nfr
        }

        total_results.append(enhanced_results)

        with open(f'PATH_TO_RETREIVED_RESULTS', 'w') as outfile:
            json.dump(total_results, outfile)


def prompt_construction(user_query, user_persona, user_stories, domain_knowledges, backgrounds):
    total_prompt = []
    for u_q, u_s, d_k, b_g in zip(user_query, user_stories, domain_knowledges, backgrounds):
        single_prompt = []
        for single_us, single_dk, single_bg in zip(u_s, d_k, b_g):
            prompt = f"""
            You are analyzing a software system to infer its expected behavior.

            Below is the structured context about the software:

            - **User Input (current query):**
              {u_q}

            - **User Persona:**
              {user_persona}

            - **Relevant User Story:**
              {single_us}

            - **Domain Knowledge:**
              {single_dk}

            - **Background Information:**
              {single_bg}

            Using all the above information, describe **the expected behavior of the software** — 
            that is, how the system should respond or act according to the user's input and the provided context.

            Please ensure your answer:
            1. Integrates details from the user persona, user story, and domain knowledge.
            2. Reflects a realistic and contextually consistent system behavior.
            3. Is clear, specific, and phrased as a software response or action description.
            """
            single_prompt.append(prompt)
        total_prompt.append(single_prompt)
    return total_prompt

if __name__ == '__main__':
    with open('PATH_TO_GENERATED_INTENTIONS') as jsonfile:
        total_foods = json.load(jsonfile)
    user_query = ["This user input an image of the nutrition list of an eating item. The food contain: " + str(food['UserSituation']['ingredients']) for food in total_foods][n:(n+125)]
    user_situation = [str(food['UserSituation']['context'] + food['UserSituation']['concern']) for food in total_foods]

    do_retrival(user_query, user_situation, n)

