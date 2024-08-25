import os
from typing import Dict
import torch

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

llama2 = {
    "model_name": "Llama-2-7b-chat-hf",
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
}

llama3 = {
    "model_name": "Meta-Llama-3-8B",
    "model_id": "meta-llama/Llama-2-7b-chat-hf",
}

llama_config = {
    'llama2' : llama2,
    'llama3' : llama3,
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "do_sample": False,
    "max_input_length": 4096,
    "SEED": 42,
}

zephyr_config = {
    "model_name": "stablelm-zephyr-3b",
    "model_id": "stabilityai/stablelm-zephyr-3b",
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "do_sample": False,
    "max_input_length": 4096,
    "SEED": 42,
}

t5_xl = {
    "model_name": "flan-t5-xl",
    "model_id": "google/flan-t5-xl"
}

t5_large = {
    "model_name": "flan-t5-large",
    "model_id": "google/flan-t5-large"
}

flan_t5_config = {
    't5_xl' : t5_xl,
    't5_large' : t5_large,
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "do_sample": False,
    "max_input_length": 4096,
    "SEED": 42,
}


prompt_rerank = "The following are docs in format id : text \n {DOCS} \n are related to query \n {QUERY} \n Instruction: Select up to k documents for answering the given question, the output containt only id of docs \n ANSWER: \n"
#prompt_rerank = "The following are docs related to query \n {QUERY} \n doc : \n  {DOC} Instruction: give me the numerical relevance score of the document in relation to the query between 1 and 0 such that 1 is the maximum and 0 is the minimum. \n ANSWER: \n "

# a revoir
retrieval_config = {
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    "experiment_name": "retrieval",
    "experiment_path": f"{ROOT_PATH}/results/",
    "results_file": "retrieval_user_query.csv",
    "query_gen_results_file": "generated_queries_4shot_4q_rerank.csv",
    "generated_queries_file": f"{ROOT_PATH}/results/zephyr_zs_query_generation/generated_queries_4shot_4q.csv",
    "posthoc_retrieval_file": f"{ROOT_PATH}/results/G/answer_generation_G.csv",
    "query_aggregation": "rerank",  # can be : "rerank",  "seperate_queries", vote, sort, simple, summed_vote, mean_vote, combSum
    "filter_queries": False,
}

# a revoir 
prompts_config = {
    "prompt_with_gold": {
        "system": "You are an assistant that provides answers and the source of the answer. I will give a question and several context texts about the question. Choose the context texts that are most relevant to the questions and based on them, give a short answer to the question. You must provide in-line citations to each statement in the answer from the context. The citations should appear as numbers within brackets [] such as [1], [2] based on the given contexts. A statement may need to be supported by multiple contexts and should then be cited as [1] [2].",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_1": {
        "system": "You are an assistant that provides - and the source of the answer. I will give a question and several context texts about the question. Choose the context texts that are most relevant to the questions and based on them, give a short answer to the question. You must provide in-line citations to each statement in the answer from the context. The citations should appear as numbers within brackets [] such as [1], [2] based on the given contexts. A statement may need to be supported by multiple contexts and should then be cited as [1] [2].",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_2": {
        "system": "You are an assistant who provides answers based on the sources I give you and you pre-know, I will ask a question and provide several contextual texts relating to the question. You must choose the most relevant contextual texts in relation to the question and, based on these, give a short answer to the question. You must provide embedded quotations for each statement in the answer, drawn from the contexts. Quotations should appear as numbers in square brackets [ ] such as [1], [2] depending on the contexts given. A statement may need to be supported by more than one context and should then be quoted as [1] [2], the form of each statement is a sentence ending in a full stop followed by the quotation(s).",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_3": {
        "system": "You are an assistant who provides answers based on the sources I give you and you pre-know, I will ask a question and provide several contextual texts relating to the question. You must choose the most relevant contextual texts in relation to the question and, based on these, give a short answer to the question. You must provide embedded quotations for each statement in the answer, drawn from the contexts. Quotations should appear as numbers in square brackets [ ] such as [1], [2] depending on the contexts given. A statement may need to be supported by more than one context and should then be quoted as [1] [2], the form of each statement is a sentence ending in a full stop followed by the quotation(s) like this 'Air pollution increases the risk of cardiovascular diseases [1].'",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_4": {
        "system": "You are an assistant who provides answers based on the sources I give you, I will ask a question and provide several contextual texts relating to the question. You must choose the most relevant contextual texts in relation to the question and, based on these, give a one short paragraph to the question. You must provide embedded quotations for each statement in the answer, drawn from the contexts. Quotations should appear as numbers in square brackets [ ] such as [1], [2] depending on the contexts given in CONTEXTS part. A statement may need to be supported by more than one context and should then be quoted as [1] [2]. each sentence in the paragraph is a statement that must end with the quotations and a full stop",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_5": {
        "system": "You are an assistant who provides answers based on the sources I give you, I will ask a question and provide several contextual texts relating to the question. You must choose the most relevant contextual texts in relation to the question and, based on these, give a one short paragraph as answer to the question. You must provide embedded quotations for each statement in the answer, drawn from the contexts. Quotations should appear as numbers in square brackets [ ] such as [1], [2] depending on the contexts given in CONTEXTS part only. A statement may need to be supported by more than one context and should then be quoted as [1] [2]. each sentence in the paragraph is a statement that must end with the quotations follow-up by point, this is a example of a statement to follow 'Air pollution increases the risk of cardiovascular diseases [1].'",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_6": {
        "system": "You are an assistant who provides answers based on the sources I give you, I will ask a question and provide several contextual texts relating to the question. You must choose the most relevant contextual texts in relation to the question and, based on these, give a one short paragraph as answer to the question composed of a set of statments only. Quotations should appear as numbers in square brackets [ ] such as [1], [2] depending on the contexts given in CONTEXTS part only. A statement may need to be supported by more than one context and should then be quoted as [1] [2]. each statement in the paragraph is a sentence that must end with the quotations of the CONTEXTS follow-up by point, this is a example of a statement to follow 'Air pollution increases the risk of cardiovascular diseases [1].'",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_7": {
    "system": "You are an assistant who provides answers based on the sources I give you. I will ask a question and provide several contextual texts relating to the question. You must choose the most relevant contextual texts in relation to the question and, based on these, give a one-paragraph answer composed of a set of statements only. Each statement must be a complete sentence and end with the relevant context number in square brackets [ ], followed by a point. For example, 'Air pollution increases the risk of cardiovascular diseases [1].' If a statement is supported by more than one context, include all relevant numbers in the brackets [1][2].",
    "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_8": {
    "system": "You are an assistant who provides answers based on the sources I give you. I will ask a question and provide several contextual texts relating to the question. You must choose the most relevant contextual texts in relation to the question and, based on these, give a one-paragraph answer composed of a set of statements only. Each statement must be a complete sentence and end with the relevant context number in square brackets [ ], followed by a point. If a statement is supported by more than one context, include all relevant numbers in the brackets [1][2], the context number should be on in the context section who has forme of number with brackets.\n here's an example to follow : QUESTION: What are the effects of deforestation? \n\n CONTEXTS:\n [1] Deforestation contributes to climate change by increasing greenhouse gas emissions.\n [2] The loss of trees leads to a decrease in biodiversity as habitats are destroyed.\n [3] Deforestation can result in soil erosion, which negatively impacts agriculture. \n\n ANSWER:\n Deforestation contributes to climate change by increasing greenhouse gas emissions [1]. The loss of trees leads to a decrease in biodiversity as habitats are destroyed [2]. Deforestation can result in soil erosion, which negatively impacts agriculture [3].",
    "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_9": {
    "system": "You are an assistant who provides answers based on the sources I give you and your prior knowledge, the answer must be a single paragraph made up of statements such as Each statement must be a complete sentence and end with the relevant context number in square brackets [ ], followed by a full stop. If a statement is supported by more than one context, indicate all the relevant numbers in square brackets [1][2], these numbers must be only one of the contexts in the context section provided.\n here's an example to follow : QUESTION: What are the effects of deforestation? \n\n CONTEXTS:\n [1] Deforestation contributes to climate change by increasing greenhouse gas emissions.\n [2] The loss of trees leads to a decrease in biodiversity as habitats are destroyed.\n [3] Deforestation can result in soil erosion, which negatively impacts agriculture. \n\n ANSWER:\n Deforestation contributes to climate change by increasing greenhouse gas emissions [1]. The loss of trees leads to a decrease in biodiversity as habitats are destroyed [2]. Deforestation can result in soil erosion, which negatively impacts agriculture [3].",
    "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_10": {
    "system": "You are an assistant who provides answers based on the sources I give you and your prior knowledge, the answer must be a single paragraph made up of statements such as Each statement must be a complete sentence and end with the relevant context number in square brackets [ ] or not, followed by a full stop. If a statement is supported by more than one context, indicate all the relevant numbers in square brackets [1][2], these numbers must be only one of the contexts in the context section provided only. \n here's an example to follow : QUESTION: What are the effects of deforestation? \n\n CONTEXTS:\n [1] Deforestation contributes to climate change by increasing greenhouse gas emissions.\n [2] The loss of trees leads to a decrease in biodiversity as habitats are destroyed. \n\n ANSWER:\n Deforestation contributes to climate change by increasing greenhouse gas emissions [1]. The loss of trees leads to a decrease in biodiversity as habitats are destroyed [2]. Deforestation can result in soil erosion, which negatively impacts agriculture .",
    "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_11": {
        "system": "You are an assistant that provides - the answer in form of one paragraph composed in multiple statment. I will give a question and several context texts about the question. Choose the context texts that are most relevant to the questions and based on them, give a short answer to the question. You must provide citations to each statement in the answer from the context only. The citations should appear as numbers within brackets [] such as [1], [2] based on the given contexts. If a statement may need to be supported by multiple contexts and should then be cited as [1] [2]. \n here's an example to follow : QUESTION: What are the effects of deforestation? \n\n CONTEXTS:\n [1] Deforestation contributes to climate change by increasing greenhouse gas emissions.\n [2] The loss of trees leads to a decrease in biodiversity as habitats are destroyed. \n\n ANSWER:\n Deforestation contributes to climate change by increasing greenhouse gas emissions [1]. The loss of trees leads to a decrease in biodiversity as habitats are destroyed [2]. Deforestation can result in soil erosion, which negatively impacts agriculture .\n\n QUESTION: What are the health impacts of a sedentary lifestyle? \n\n CONTEXTS:\n [1] A sedentary lifestyle increases the risk of cardiovascular diseases.\n [2] Prolonged periods of sitting are linked to higher rates of obesity.\n [3] Physical inactivity can lead to mental health issues like depression and anxiety. \n\n ANSWER:  A sedentary lifestyle increases the risk of cardiovascular diseases [1]. It is also linked to higher rates of obesity due to prolonged periods of sitting [2]. Additionally, physical inactivity can contribute to mental health issues like depression and anxiety [3].",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_12": {
        "system": "You are an assistant that provides answers based on provided contextual information. You will be given a question and several context texts related to that question. Your task is to choose the contexquestiont texts that are most relevant to the question and, based on them, provide a concise answer of statement. Every statement in your answer must include citation or not it's depend if he can be supported, using numbers within brackets [] corresponding to the provided contexts. All citations must be directly drawn from the context section provided only.",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompt_13": {
        "system": "You are an assistant that provides answers based on provided contextual information. You will be given a question and several context texts related to that question. Your task is to choose the context texts that are most relevant to the question and, based on them, provide a concise answer. Every statement in your answer must include citation or not, using numbers within brackets [] corresponding to the provided contexts. All citations number must be directly drawn from the context section provided only. IF a statement may need to be supported by multiple contexts and should then be cited as [1] [2].\n\n QUESTION: What are the effects of deforestation? \n\n CONTEXTS:\n [1] Deforestation contributes to climate change by increasing greenhouse gas emissions.\n [2] The loss of trees leads to a decrease in biodiversity as habitats are destroyed. \n\n ANSWER: Deforestation contributes to climate change by increasing greenhouse gas emissions [1]. The loss of trees leads to a decrease in biodiversity as habitats are destroyed will[2]. Deforestation can also result in soil erosion, which negatively impacts agriculture.",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "prompot_without_context": {
        "system": "You are an assistant that provides answers. I will give you a question and based on your knowledge, give a very short answer to the question.",
        "user": "QUESTION: {query} \n\n ANSWER:",
    },
    "prompt_without_citation": {
        "system": "You are an assistant that provides answers. I will give a question and several context texts about the question. Choose the context texts that are most relevant to the questions and based on them, give a brief answer to the question.",
        "user": "QUESTION: {query} \n\n CONTEXTS:\n {context} \n\n ANSWER:",
    },
    "query_gen_prompt": {
        "system": "You are an assistant that helps the user with their search. I will give you a question, and based on this question, you will suggest other specific queries that help retrieve documents that contain the answer. Only generate your suggested queries without explanation. The maximum number of queries is 4.",  # "You are an assistant that helps the user with their search. I will give you a question and its answer, based on this question and the answer, you will suggest other specific queries that help retrieve documents that contain the answer. The maximum number of queries is 4.",
        "user_with_answer": "QUESTION: {query} \n\n ANSWER:\n {answer} \n\n SUGGESTED QUERIES:",
        "user": "QUESTION: {query} \n\n SUGGESTED QUERIES:",
    },
}

# a revoir
exp_zephyr_query_gen_fewshots = {
    "experiment_name": "_zs_query_generation",
    "experiment_path": f"{ROOT_PATH}/results/",
    "results_file": "generated_queries_4shot_4q.csv",
    "config_file": "generated_queries_4shot_4q.json",
    "setting": "fewshot",  # zeroshot
    "query_gen_prompt": {
        "system": "You are an assistant that helps the user with their search. I will give you a question, based on the possible answer of the question you will provide queries that will help find documents that support it. Only generate your suggested queries without explanation. The maximum number of queries is {nb_queries}",  # .
        "user_with_answer": "QUESTION: {query} \n\n ANSWER:\n {answer} \n\n SUGGESTED QUERIES:",
        "user": "QUESTION: {query} \n\n SUGGESTED QUERIES:",
    },
    "include_answer": False,
    "nb_queries_to_generate": 4,
    "nb_shots": 4,
    "fewshot_examples": [
        {
            "user_query": "Why does milk need to be pasteurized?",
            "generated_queries": [
                "How does pasteurization work to make milk safer?",
                "What are the arguments that support milk pasteurization?"
                "What is the purpose of milk pasteurization?",
                "Adoption of milk pasteurization in developed countries",
                "What are the differences between pasteurization and sterilization in milk processing?",
                "United States raw milk debate",
            ],
        },
        {
            "user_query": "What is the largest species of rodent?",
            "generated_queries": [
                "Is there a rodent bigger than a capybara?",
                "How big are giant hutia vs capybara?",
                "Comparison of capybara with other rodent species",
                "the world's heaviest rodent species",
                "Sizes of common rodents",
            ],
        },
        {
            "user_query": "What was the first animal to be domesticated by humans?",
            "generated_queries": [
                "Examples of early domesticated animals",
                "Commensal pathway of dog domestication",
                "First animal domesticated by humans",
                "When did the domestication process of animals first start?",
                "Is there evidence supporting the domestication of dogs?",
                "What were the first steps of animal domestication?",
                "What started the dog domestication process?",
            ],
        },
        {
            "user_query": "When is All Saints Day?",
            "generated_queries": [
                "What do Catholics do on All Saints Day?",
                "Observance of All Saints Day by non-Catholic Christians",
                "Solemnity of All Saints Day and its transfer to Sundays",
                "What is the difference between Day of the Dead and All Saints Day?",
                "Is All Saints Day a Catholic holy day?",
                "Do Protestants celebrate All Saints Day?",
            ],
        },
        {
            "user_query": "Who won the battle of Trafalgar?",
            "generated_queries": [
                "Napoleonic Wars duration after the Battle of Trafalgar",
                "How did the victory at the Battle of Trafalgar impact the Napoleonic Wars?",
                "Was Napoleon defeated at Trafalgar?",
                "What were the strategic implications of the British victory at the Battle of Trafalgar?",
                "Impact of the Battle of Trafalgar on British naval supremacy",
                "Paintings commemorating the Battle of Trafalgar",
            ],
        },
        {
            "user_query": "What was the last Confederate victory in the Civil War?",
            "generated_queries": [
                "What was the final major battle for the Confederacy?",
                "When was the Battle of Natural Bridge fought?",
                "Battle of Plymouth as an earlier Confederate victory",  ### was not used in first tests
                "Battle of Palmito Ranch",
                "Was Battle of Palmito Ranch the final battle of the American Civil War?",
                "Comparison of Battle of Palmito Ranch with other Confederate victories",
            ],
        },
    ],
}

llms_config = {"zephyr": zephyr_config, "llama2": llama_config, "llama3": llama_config, 't5_large': flan_t5_config, 't5_xl' : flan_t5_config }

evaluation = {
    "reference_column": "gold_truth",
    "prediction_column": "generated_text",
    "keep_citation": False,
    "cache_dir": f"{ROOT_PATH}/models_cache/",
    #"results_file": f"{ROOT_PATH}/results/llms/zephyr_zs_hagrid_ctxt_citing/zephyr_zs_answer_generation_without_context",
}

architectures_config = {
    "G": {
        "use_retrieved": False,
        "retrieved_passages_file": None,
        "use_context": False,
        "nb_passages": None,
        "citation": False,
        "experiment_name": "G",
        "experiment_path": f"{ROOT_PATH}/results/",
        "results_file": "answer_generation_G.csv",
        "config_file": "answer_generation_G.json",
    },
    "RTG-gold": {
        "use_retrieved": True, # False
        "retrieved_passages_file": None,
        "use_context": True,
        "nb_passages": None,
        "citation": True,
        "experiment_name": "RTG_gold",
        "experiment_path": f"{ROOT_PATH}/results/",
        "results_file": "answer_generation_RTG_gold_passages.csv",
        "config_file": "answer_generation_RTG_gold_passages.json",
    },
    "RTG-vanilla": {
        "use_retrieved": True,
        "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/retrieval_user_query.csv",
        "use_context": True,
        "nb_passages": 3,
        "citation": True,
        "experiment_name": "RTG_vanilla",
        "experiment_path": f"{ROOT_PATH}/results/",
        "results_file": "generation_RTG_vanilla_2_passages.csv",
        "config_file": "generation_RTG_vanilla_2_passages.json",
    },
    "RTG-query-gen": {
        "use_retrieved": True,
        "retrieved_passages_file": f"{ROOT_PATH}/results/retrieval/generated_queries_4shot_4q_rerank.csv",  # devMiracl_results_MonoT5_BM500_20_normal_corpus.csv",  # generated_queries_4shot_4q_lbre_nb_example_pmpt2_desc_seperate
        "use_context": True,
        "nb_passages": 3,
        "citation": True,
        "experiment_name": "RTG_generated_queries",
        "experiment_path": f"{ROOT_PATH}/results/",
        "results_file": "answer_generation_RTG_gen_queries_4q_4shots_rerank_5_passages.csv",
        "config_file": "answer_generation_RTG_gen_queries_4q_4shots_rerank_5_passages.json",
    },
}

dataset_config = {"comp24" : 'topics.rag24.test.txt',
    "MS_MARCO_Dev2": "topics.msmarco-v2-doc.dev2.txt", "MS_MARCO_Dev": "topics.msmarco-v2-doc.dev.txt",
                  "TREC_DL_2023" : "topics.dl23.txt", "TREC_DL_2022" : "topics.dl22.txt", "TREC_DL_2021" : "topics.dl21.txt"
                 }

corpus_config = {'complete': 'msmarco-v2.1-doc','segmented':"msmarco-v2.1-doc-segmented"}

CONFIG: Dict = {
    "architectures": architectures_config,
    "langauge_model": llms_config,
    "dataset": dataset_config,
    "data_path": "DATA",  #None
    "prompts": prompts_config,
    "retrieval": retrieval_config,
    "query_generation": exp_zephyr_query_gen_fewshots,
    "results_columns": {
        "prediction": "generated_text",
        "reference": "gold_truth",
    },
    "evaluation" : evaluation,
    "column_names" : {
        "passages" : "quotes",
        "query" : "query"
        },

    "passage_column" : {
        "topic_id" : 0,
        "docid" : 2,
        "rank" : 3,
        "score" : 4,

    },

    'corpus' : corpus_config,

    'prompt_rerank' : prompt_rerank
}
