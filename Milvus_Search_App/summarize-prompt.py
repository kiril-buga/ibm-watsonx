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

# 3. Define summarize prompt template
summarize_template = PromptTemplate(
    name="Document Zusammenfassung",
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    input_prefix="Human:",
    instruction="""### Anweisung:
Du bist ein hilfreicher Assistent, der Fragen **ausschliesslich** mithilfe der
im CONTEXT-Block gelieferten Dokumentpassagen beantwortet.

• Nutze nur den CONTEXT – erfinde oder rate nichts dazu.  
• Fassen Sie den Inhalt der folgenden Kategorien zusammen, falls verfügbar:
1. Kapital
2. Prämie
3. Versicherungsschutz und Leistungen
4. Allgemeine Geschäftsbedingungen

• Ist die Antwort vorhanden, zitiere oder paraphrasiere sie und nenne die   Quelle als *(Datei, S. Seite)*.  
• Liefern die Passagen nur Teilinformationen, erläutere kurz deine Schlussfolge.  
• Kann die Frage mit dem CONTEXT nicht beantwortet werden, antworte exakt:  
  „Ich weiss es nicht auf Grundlage der bereitgestellten Dokumente.“  
• Wenn sinnvoll, beginne mit einer 1-Satz-Zusammenfassung vor der eigentlichen
  Antwort.  
• **Antworte in derselben Sprache wie die Benutzerfrage.**.
""",
    output_prefix="Assistant:",
    input_text=r"""
    ### CONTEXT
    {context}

    Datei: {file_name} | Seite: {page_number}
    Produkt: {product_name} {product_month}.{product_year}
    Unternehmen: {company_entity} | Kapitel: {chapter}

    ### User Query
    {user_query}

    ### AUFGABE
    Fassen Sie die Police anhand der folgenden Aspekte zusammen:
    1. Kapital
    2. Prämie
    3. Versicherungsschutz und Leistungen
    4. Allgemeine Geschäftsbedingungen

    Bitte geben Sie für jeden Teil die Dokument- und Seitenzahlen an.
    """,
    input_variables=[
        "context", "user_query", "file_name", "chapter",
        "company_entity", "page_number", "product_name",
        "product_month", "product_year"
    ],
    model_params={
        "decoding_method": "greedy",
        "repetition_penalty": 1,
        "temperature": 0.0,
        "max_new_tokens": 1024
    }
)

# 4. Store template
stored_template = prompt_mgr.store_prompt(summarize_template)
print(f"Prompt Template ID: {stored_template.prompt_id}")

# 5. Deploy template
from ibm_watsonx_ai import APIClient
client = APIClient(credentials)
client.set.default_space(space_id)

deployment = client.deployments.create(
    artifact_id=stored_template.prompt_id,
    meta_props={
        client.deployments.ConfigurationMetaNames.NAME: "Document Summarize Deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {},
        client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    }
)
print(f"Deployment ID: {deployment['metadata']['id']}")
# Prompt Template ID: 35fe0fe7-090c-44ed-88b2-3504062e8956


# ######################################################################################

# Synchronous deployment creation for id: '35fe0fe7-090c-44ed-88b2-3504062e8956' started

# ######################################################################################


# initializing
# Note: online_url and serving_urls are deprecated and will be removed in a future release. Use inference instead.

# ready


# -----------------------------------------------------------------------------------------------
# Successfully finished deployment creation, deployment_id='3dcda8c7-4cea-41dd-80b3-1884fb2d01c8'
# -----------------------------------------------------------------------------------------------


# Deployment ID: 3dcda8c7-4cea-41dd-80b3-1884fb2d01c8
 