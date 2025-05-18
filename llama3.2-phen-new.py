print("...Running Python file...")
from ollama import chat
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json

print("...Imported Libraries...")

#df = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/Old AD notes/AlzheimerNotes.csv')
#results_dir = '/storage1/fs1/aditigupta/Active/Will/Local_LLM_AD_Results/'
df = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/mdc_progress_notes_new.csv')
results_dir = '/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling'

class Phenotype(BaseModel):
    repeat: str
    misplaces_objects: str
    famhx: str


system_prompt = (
    "Assistant is an intelligent chatbot designed to extract and determine specific "
    "clinical information from a provided clinical note."
)

user_prompt = (
    "1. Determine if the patient repeats questions or statements. Answer only \"yes\" or \"no\".\n 2. Determine if the patient misplaces objects. Answer only \"yes\" or \"no\". 3. Determine if the patient has a family history of dementia. Answer only \"yes\" or \"no\". \n Please provide the output in JSON format. Here is the note: {}"
)

system_json_prompt = (
    "Assistant is an intelligent chatbot designed to take in input and format it into JSON format."
)

user_json_prompt = "Here is the text to be formatted into JSON: {}"

models_list = ['llama3.2-120k']

def local_llm_phenotype_ad_notes(df, models_list, system_prompt, user_prompt, system_json_prompt, user_json_prompt):
    """
    Run Local LLMs using Llama to phenotype AD Notes.
    """
    # Initialize output dataframe with correct dimension and column names
    output_df = pd.DataFrame({
        'EPIC_MRN': np.zeros(len(df)),
        'NOTE_ID': np.zeros(len(df)),
        'repeat': np.zeros(len(df)),
        'misplaces_objects': np.zeros(len(df)),
        'famhx': np.zeros(len(df)),
    })
    print("running loop")

    # Run LLMs on clinical notes
    for model in models_list:
        phenotypes_list = []

        for i in range(5829, len(df)):
            note = df.iloc[i, -1]
            response = chat(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt.format(note)}
                ],
                model=model,
                format=Phenotype.model_json_schema(),
            )

            phenotypes = Phenotype.model_validate_json(response['message']['content'])
            print(phenotypes)
            phenotypes_list.append(phenotypes)
            print(i)

            response_json = chat(
                messages=[
                    {'role': 'system', 'content': system_json_prompt},
                    {'role': 'user', 'content': user_json_prompt.format(phenotypes)}
                ],
                model=model,
                format=Phenotype.model_json_schema(),
            )

            print(response_json['message']['content'])
            phen_json_string = response_json['message']['content']
            phen_json = json.loads(phen_json_string)

            # Access each component of JSON prompt
            epic_mrn = df.iloc[i, 2]  # Check this
            note_id = df.iloc[i, 3]
            repeat_outcome = phen_json['repeat']
            misplace_outcome = phen_json['misplaces_objects']
            famhx_outcome = phen_json['famhx']

            # Write to dataframe
            output_df.iloc[i, 0] = epic_mrn
            output_df.iloc[i, 1] = note_id
            output_df.iloc[i, 2] = repeat_outcome
            output_df.iloc[i, 3] = misplace_outcome
            output_df.iloc[i, 4] = famhx_outcome

            # Save dataframe
            print("Saving to file")
            file_path = f"{results_dir}/results_{model}_llama_repeat_misplace_famhx_new3.csv"
            output_df.to_csv(file_path)

local_llm_phenotype_ad_notes(df, models_list, system_prompt, user_prompt, system_json_prompt, user_json_prompt)
