import re
import os
import shutil
from datetime import datetime

import pandas as pd
import polars as pl
import replicate
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from uptrain import GuidelineAdherence

load_dotenv()

AA_MODEL_NAME = "luminous-supreme-control"
# Guideline name used in the Guideline Adherence check
GUIDELINE_NAME = "Strict_Context"
RESULTS_DIR = "./results/"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")

# Azure deployments:
GPT_35_TURBO_16K = "gpt-35-turbo-16k-deployment"
GPT_4 = "gpt4"

SYSTEM_PROMPT = """Please generate an ANSWER that strictly adheres to the given CONTEXT and accurately addresses the QUESTION asked, without adding any of the model's own information. If the required information is not found in the CONTEXT, respond in German with: 'Ihre Anfrage kann nicht mit den bereitgestellten Daten beantwortet werden. Bitte erläutern Sie Ihre Anfrage genauer oder geben Sie weitere Informationen an, falls notwendig'. Avoid references to previous outputs of the model. The answer should be based solely on the provided CONTEXT. If the QUESTION does not relate to a health-related topic, briefly explain why the query cannot be answered and recommend a more precise formulation or additional information. Always answer in German.

CONTEXT:
{context}"""

COHERE_MESSAGE = """Please generate an ANSWER that strictly adheres to the provided DOCUMENTS and accurately addresses the QUESTION asked, without adding any of the model's own information. If the required information is not found in the DOCUMENTS, respond with: 'Ihre Anfrage kann nicht mit den bereitgestellten Daten beantwortet werden'. The answer should be based solely on the provided DOCUMENTS. If the QUESTION does not directly relate to a health-related topic or is not clearly answerable, briefly explain why the query cannot be answered and recommend a more precise formulation or additional information.

QUESTION:
{question}"""

SYSTEM_PROMPT_GERMAN = """### INSTRUKTIONEN
Generiere bitte eine ANTWORT, die sich strikt an den gegebenen KONTEXT hält und präzise auf die gestellte FRAGE antwortet, ohne eigene Informationen des Modells hinzuzufügen. Falls die benötigte Information nicht im KONTEXT zu finden ist, antworte mit: 'Ihre Anfrage kann nicht mit den bereitgestellten Daten beantwortet werden. Bitte erläutern Sie Ihre Anfrage genauer oder geben Sie weitere Informationen an, falls notwendig.'. Vermeide Bezüge auf vorherige Ausgaben des Modells. Die Antwort soll auf dem bereitgestellten KONTEXT basieren. Sollte die FRAGE nicht direkt einem gesundheitsbezogenen Thema zuzuordnen sein oder nicht klar zu beantworten sein, erkläre kurz, warum die Anfrage nicht beantwortet werden kann und empfehle eine genauere Formulierung oder zusätzliche Informationen.

### KONTEXT
{context}"""

# Guideline adherence check
GUIDELINE = "The response must strictly adhere to the provided context and not introduce external information. If the necessary information is absent from the context, respond with: 'Ihre Anfrage kann nicht mit den bereitgestellten Daten beantwortet werden. Bitte erläutern Sie Ihre Anfrage genauer oder geben Sie weitere Informationen an, falls notwendig.'. Should the question fall outside the health-related jurisdiction of the Landesgesundheitsamt Niedersachsen, it means the query is beyond the health-related scope and shouldn't be answered."


def get_experiment_file_path(experiment_name, extension):
    filename = f"{experiment_name.replace(' ', '_').replace('-', '_').lower()}_experiment.{extension}"
    return os.path.join(RESULTS_DIR, filename)


def read_dataset(path):
    ensure_directory_exists(RESULTS_DIR)
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified dataset path does not exist: {path}")
    # return pd.read_json(path, lines=True)
    return pl.read_ndjson(path)


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def initialize_azure_openai_client():
    return AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_API_BASE
    )


def initialize_openai_client(api_key, base_url):
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def get_response(client, row, model):
    question = row['question'][0]
    context = row['context'][0]
    message = SYSTEM_PROMPT.format(context=context)

    if "azureai/" in model:
        api_model, returned_model = model.split('/')
    else:
        api_model = returned_model = model

    print(f"___Question: {question}"
          # f"___Message: {message}"
          )

    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content": message},
            {"role": "user", "content": question},
            # {"role": "assistant", "content": "Example answer"},
            # {"role": "user", "content": "First question/message for the model to actually respond to."}
        ]
    ).choices[0].message.content
    print(f"___Response: {response}\n")

    return {
        'question': question,
        'context': context,
        'response': response,
        'ground_truth': row['ground_truth'][0],
        'model': returned_model
    }


CONTEXT_PROMPT_old = """Generate an ANSWER that strictly adheres to the provided DOCUMENTS and addresses the QUESTION asked precisely, without adding any information from the model itself. If the required information is not found in the DOCUMENTS, respond with: 'Ihre Anfrage kann nicht mit den bereitgestellten Daten beantwortet werden'. Avoid references to previous outputs of the model. The response should be based solely on the provided DOCUMENTS. If the QUESTION is not directly related to a health topic or is not clearly answerable, briefly explain why the query cannot be answered and suggest a more specific formulation or additional details. 
DON'T include the prefatory phrase "Answer:" in your response!

QUESTION
{question}

DOCUMENTS
{context}
"""
CONTEXT_PROMPT_old_chat = """Generate an concise ANSWER that strictly adheres to the provided DOCUMENTS and addresses the QUESTION asked precisely, without adding any information from the model itself. If the required information is not found in the DOCUMENTS, respond with: 'Ihre Anfrage kann nicht mit den bereitgestellten Daten beantwortet werden'. Avoid references to previous outputs of the model. The response should be based solely on the provided DOCUMENTS. If the QUESTION is not directly related to a health topic or is not clearly answerable, briefly explain why the query cannot be answered and suggest a more specific formulation or additional details. 
DON'T include the prefatory phrase "Answer:" in your response!
Please always answer in German.

DOCUMENTS
{context}
"""


def get_response_vllm(client, row, model):
    question = row['question'][0]
    context = row['context'][0]
    # prompt = CONTEXT_PROMPT_old.format(context=context, question=question)
    # prompt = CONTEXT_PROMPT_old_chat.format(context=context)
    prompt = SYSTEM_PROMPT.format(context=context)
    stop_tokens = ["\n\n\n"]

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]

    if model == "meta-llama/Meta-Llama-3-8B-Instruct":
        # prompt = SYSTEM_PROMPT_GERMAN.format(context=context)
        stop_tokens = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
    elif "openchat" in model:
        # prompt = SYSTEM_PROMPT.format(context=context)
        stop_tokens = ["</s>", "GPT4", "<|end_of_turn|>"]
    elif "mistral" in model or "mixtral" in model:
        # Mistral instruct and Mixtral instruct doesn't accept system prompt.
        prompt = f"{prompt}\nQUESTION:\n{question}"
        messages = [{"role": "user", "content": prompt}]
    elif "WestLake" in model:
        stop_tokens = ["/INST"]

    print(f"___Question: {question}")

    # completion = client.completions.create(model=model,
    #                                        prompt=prompt,
    #                                        temperature=0.1,
    #                                        max_tokens=200)
    # response = completion.choices[0].text

    completion = client.chat.completions.create(
        model=model,
        max_tokens=500,
        temperature=0.1,
        stop=stop_tokens,
        messages=messages,
    )
    # print("Completion result:", completion)
    response = completion.choices[0].message.content
    print(f"___Response: {response}\n")

    return {
        'question': question,
        'context': context,
        'response': response,
        'ground_truth': row['ground_truth'][0],
        'model': model
    }


REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


def get_response_replicate(row, model_name):
    # https://replicate.com/meta/meta-llama-3-8b-instruct/api/learn-more
    # https://replicate.com/kcaverly/nous-capybara-34b-gguf/api
    question = row['question'][0]
    context = row['context'][0]
    system_prompt = SYSTEM_PROMPT.format(context=context)
    prompt = f"{system_prompt}\nQUESTION:\n{question}"

    # if smth like "kcaverly/nous-capybara-34b-gguf:6b9b7741e719899f26571567b892a8900d0b517bfadee3997f5d477897d10eef"
    if ":" in model_name:
        returned_model = model_name.split(':')[0]
    else:
        returned_model = model_name
    print(f"___Question: {question}")
    input = {
        "prompt": prompt,
        "temperature": 0.1,
    }

    output = replicate.run(
        model_name,
        input=input
    )

    response = "".join(output)
    print(f"___Response: {response}\n")
    return {'question': question, 'context': context, 'response': response, 'ground_truth': row['ground_truth'][0],
            'model': returned_model}


def split_documents(context):
    # need to split the context by document number headings like "[Doc Nr. 1]"
    docs = re.split(r'\[Doc Nr\. \d+\]', context)
    titles = re.findall(r'(\[Doc Nr\. \d+\])', context)

    documents = []
    for title, snippet in zip(titles, docs[1:]):
        documents.append({"title": title, "snippet": snippet.strip()})
    return documents


def get_response_cohere(client, row, model):
    question = row['question'][0]
    context = row['context'][0]
    documents = split_documents(context)
    message = COHERE_MESSAGE.format(question=question)

    if "azureai/" in model:
        api_model, returned_model = model.split('/')
    else:
        api_model = returned_model = model

    print(f"message: {message}, \n"
          f"documents:\n {documents}")

    response = client.chat(
        message=message,
        documents=documents
    )

    return {
        'question': question,
        'context': context,
        'response': response.text,
        'ground_truth': row['ground_truth'][0],
        'model': returned_model
    }


def format_response(row):
    question = row['question'][0]
    context = row['context'][0]
    response = row['response'][0]
    ground_truth = row['ground_truth'][0]
    model = AA_MODEL_NAME

    return {'question': question, 'context': context, 'response': response, 'ground_truth': ground_truth,
            'model': model}


def update_guidelines(res_guidelines, config_model_name):
    if "azureai/" in config_model_name:
        print("Azure AI model detected, removing prefix...")
        config_model_name = config_model_name.split('/')[1]
    elif ":" in config_model_name:
        config_model_name = config_model_name.split(':')[0]
    DEFAULT_SCORE = float("nan")
    DEFAULT_EXPLANATION = "No data available"
    score_name = 'score_' + GUIDELINE_NAME + '_adherence'
    explanation_name = 'explanation_' + GUIDELINE_NAME + '_adherence'

    for f in res_guidelines:
        score_key = score_name + '_model_' + config_model_name
        explanation_key = explanation_name + '_model_' + config_model_name

        if score_name in f:
            f[score_key] = f.pop(score_name)
        else:
            f[score_key] = DEFAULT_SCORE

        if explanation_name in f:
            f[explanation_key] = f.pop(explanation_name)
        else:
            if score_key not in f or f[score_key] == DEFAULT_SCORE:
                f[explanation_key] = DEFAULT_EXPLANATION

    return res_guidelines


def run_guideline_adherence_eval(eval_llm, data):
    return eval_llm.evaluate(
        data=data,
        checks=[GuidelineAdherence(guideline=GUIDELINE, guideline_name=GUIDELINE_NAME)]
    )


def merge_lists(base_list, update_list):
    update_dict = {item['question']: item for item in update_list if 'question' in item}

    for item in base_list:
        question = item.get('question')
        if question and question in update_dict:
            # print(f"updating with {question}")
            update_info = {key: val for key, val in update_dict[question].items() if key != 'response'}
            item.update(update_info)
    return base_list


def backup_and_save_df(df, file_path, file_type='csv'):
    backup_dir = os.path.join(os.path.dirname(file_path), 'backups')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_filename = os.path.basename(file_path) + f".backup-{timestamp}"
        backup_path = os.path.join(backup_dir, backup_filename)
        shutil.copy(file_path, backup_path)

    if file_type == 'csv':
        print(f"Saving DataFrame to CSV at: {file_path}")
        df.write_csv(file_path)
    elif file_type == 'jsonl':
        print(f"Saving DataFrame to NDJSON at: {file_path}")
        df.write_ndjson(file_path)


def display_average_scores(df):
    score_columns = [col for col in df.columns if 'score' in col]
    data_for_table = []

    for column in score_columns:
        average = df[column].drop_nans().mean()

        parts = column.split('_model_')
        # print(f"___ parts: {parts}")
        metric_name = parts[0].replace('score_', '').replace('_', ' ').capitalize()
        model_name = parts[1]
        # print(f"metric_name: {metric_name}, average: {average}")
        # print(f"model_name: {model_name}")

        data_for_table.append({
            "Model": model_name,
            "Metric": metric_name,
            "Average Score": average
        })

    results_table = pl.DataFrame(data_for_table)
    # print(data_for_table)
    return results_table
