from ibm_watsonx_ai.foundation_models.prompts import PromptTemplate, PromptTemplateManager
from ibm_watsonx_ai import Credentials

# Set your credentials and space/project ID
credentials = Credentials(
    api_key="0s6V0jgwZm9l-krqdmu2NgSe0jG0f3c8spveYgwMqEJQ",
    url="https://eu-de.ml.cloud.ibm.com"
)
space_id = "5693155d-ed73-4f99-a65f-15805c8ad662"

# 2. Create prompt template manager
prompt_mgr = PromptTemplateManager(credentials=credentials, space_id=space_id)

# 3. Define comparison prompt template
comparison_template = PromptTemplate(
    name="Document Comparison Analyst",
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    instruction="""### Anweisung:
Du bist ein hilfreicher Assistent der Textpassagen analysiert und vergleicht, die durch RAG extrahiert wurden. Der Kontext wird jeweils im Input geliefert und du als Assistenz beziehst dich einzig auf das gelieferte Material.

Die Antwort soll:
- Ausschliesslich auf den bereitgestellten Kontexten basieren.
- Unterschiede und ggf. Gemeinsamkeiten strukturiert und möglichst klar darstellen – idealerweise auch in Markdown Tabellenform.
- Deutlich machen, aus welchem Dokument (Dateiname) die jeweilige Information stammt und welchem Produktnamen es zugeordnet ist.
- Keine Inhalte erfinden oder interpretieren, sondern nur analysieren und wiedergeben, was in den Kontexten steht.

""",
    output_prefix="Assistant:",
    input_text="""KONTEXT:
Year: {year_1}
File: {file_name_1}
Document: {context_year1}

Year: {year_2}
File: {file_name_2}
Document: {context_year2}

Wie unterscheidet sich die '{product_name}' Police von {year_1} mit der von {year_2}, auf der Basis von: {user_query}?""",
    input_variables=[
        "context_year1", "context_year2",
        "year_1", "year_2",
        "product_name", "user_query",
        "file_name_1", "file_name_2"
    ],
    examples=[[
       """KONTEXT:
Year: 2021
File: file1.pdf
Document: Die Police von 2021 hat einen Zinsfuss von 0.25%.

Year: 2023
File: file2.pdf
Document: Die Police von 2023 hat einen Zinsfuss von 0.05%.

Wie unterscheidet sich die 'Absicherungsplan' Police von 2021 mit der von 2023, auf der Basis von: Zinsfuss?""",
            # Example output (markdown table)
            """### Analyse:
1. **Berechnungsgrundlage**:
   - **2021 (file1.pdf)**: Die Police von 2021 hat einen Zinsfuss von 0.25%.
   - **2023 (file2.pdf)**: Die Police von 2023 hat einen Zinsfuss von 0.05%.

| Kriterium   | 2021 (file1.pdf) | 2023 (file2.pdf) |
|-------------|------------------|------------------|
| Zinsfuss    | 0.25%            | 0.05%            |
"""
        ]
    ],
    model_params={
        "decoding_method": "greedy",
        "repetition_penalty": 1,
        "temperature": 0.0,
        "max_new_tokens": 2048
    }
)

# 4. Store template
stored_template = prompt_mgr.store_prompt(comparison_template)
print(f"Prompt Template ID: {stored_template.prompt_id}")

# 5. Deploy template
from ibm_watsonx_ai import APIClient
client = APIClient(credentials)
client.set.default_space(space_id)

deployment = client.deployments.create(
    artifact_id=stored_template.prompt_id,
    meta_props={
        client.deployments.ConfigurationMetaNames.NAME: "Document Comparison Deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {},
        client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    }
)
print(f"Deployment ID: {deployment['metadata']['id']}")