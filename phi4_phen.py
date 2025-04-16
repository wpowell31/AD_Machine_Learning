print("...Running Python file...")
from ollama import chat
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
print("...Imported Libraries...")



#df = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/Old AD notes/AlzheimerNotes.csv')
#results_dir = '/storage1/fs1/aditigupta/Active/Will/Local_LLM_AD_Results/'
df = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/ad_patient_subset_notes.csv', nrows=120)
results_dir = '/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract'

class Phenotype(BaseModel):
    misplace: str
    repeat: str
    ttau: str
    ptau: str
    famhx: str
    



system_prompt = "Assistant is an intelligent chatbot designed to extract and determine specific clinical information from a provided clinical note."
user_prompt = "Instructions: 1. Determine if the patient misplaces objects. Answer only \"yes\" or \"no\". 2. Determine if the patient repeats questions or statements. Answer only \"yes\" or \"no\". 3. From the clinical note provided, please extract the concentration of total tau (also abbreviated t-tau or tau, never p-tau). Provide only the value and units in your answer; if no units are available, just provide the value. Do not provide an explanation.\n Here are 2 examples of the output format:\n Total tau concentration: 500 pg/ml\n Total tau concentration: 600 (no units)\n If you are unable to find a total tau concentration, output only \"N/A\". 4. From the clinical note provided, please extract the concentration of phosphorylated tau (also abbreviated p-tau or phos-tau). Provide only the value and units in your answer; if no units are available, just provide the value. Do not provide an explanation.\n Here are 2 examples of the output format:\n Phosphorylated tau concentration: 80 pg/ml\n Phosphorylated tau concentration: 70 (no units)\n If you are unable to find a phosphorylated tau concentration, output only \"N/A\". 5. Determine if the patient has a family history of dementia. Answer only \"yes\" or \"no\". Please provide each piece of output in JSON format. Here is the note: {}"
system_json_prompt = "Assistant is an intelligent chatbot designed to take in input and format it into JSON format."
user_json_prompt = "Here is the text to be formatted into JSON: {}"
models_list = ['phi4-120k',]



def local_llm_phenotype_ad_notes(df, models_list, system_prompt, user_prompt, system_json_prompt, user_json_prompt):
    """
    Run Local LLMs using Llama to phenotype AD Notes.
    """
    # Initialize output dataframe with correct dimensiona and column names
    output_df = pd.DataFrame({
        'EPIC_MRN': np.zeros(len(df)),
        'misplace': np.zeros(len(df)),
        'repeat': np.zeros(len(df)),
        'ttau': np.zeros(len(df)),
        'ptau': np.zeros(len(df)),
        'famhx': np.zeros(len(df))
    })
    print("running loop")
    # Run LLMs of clinical notes
    for model in models_list:
        phenotypes_list = []
        for i in range(len(df)):
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

            # Access each components of JSON prompt
            epic_mrn = df.iloc[i,2] #check this
            misplace_outcome = phen_json['misplace']
            repeat_outcome = phen_json['repeat']
            ttau_outcome = phen_json['ttau']
            ptau_outcome = phen_json['ptau']
            famhx_outcome = phen_json['famhx']

            # write to dataframe
            output_df.iloc[i,0] = epic_mrn
            output_df.iloc[i,1] = misplace_outcome
            output_df.iloc[i,2] = repeat_outcome
            output_df.iloc[i,3] = ttau_outcome
            output_df.iloc[i,4] = ptau_outcome
            output_df.iloc[i,5] = famhx_outcome

            # save dataframe
            file_path = f"{results_dir}/results_{model}_3.csv"
            output_df.to_csv(file_path)




local_llm_phenotype_ad_notes(df, models_list, system_prompt, user_prompt, system_json_prompt, user_json_prompt)