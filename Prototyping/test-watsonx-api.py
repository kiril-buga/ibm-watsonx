import os

from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

creds = Credentials(
    api_key=os.getenv("API_KEY"),
    url="https://eu-de.ml.cloud.ibm.com",  # region root
)


def run():
    model = ModelInference(
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        credentials=creds,
        project_id=os.getenv("PROJECT_ID"),
    )
    prompt= '''Hallo, kannst du diese 2 Passagen vergleichen?
    "passage: 3.2 Definition Unfall\nUnfall ist die plötzliche, nicht beabsichtigte schädigende Einwirkung eines ungewöhnlichen äusseren Faktors auf den menschlichen Körper, die eine Beeinträchtigung der körperlichen, geistigen oder psychischen Gesundheit oder den Tod zur Folge hat.",
     "passage: 3.2 Definition Unfall\n- a) Gesundheitsschädigung durch unfreiwilliges Einatmen von plötzlich ausströmenden Gasen oder Dämpfen;\n- b) Vergiftung oder Verletzung durch unabsichtliches Einnehmen von giftigen oder ätzenden Stoffen;\n- c) Unfreiwilliges Ertrinken, Erfrieren oder Hitzschlag;\n- d) Vom Willen des Versicherten unabhängige Verbrennungen, Verbrühungen, Einwirkungen von Blitzschlag oder elektrischem Strom.",
            
    '''
    generate_params = {
        # keep generating until we really stop it
        GenParams.MAX_NEW_TOKENS: 1024,
        GenParams.STOP_SEQUENCES: [],  # ⚠️ turn off the default “\n\n”
        # optional but recommended
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.TEMPERATURE: 0,  # deterministic
    }

    print("Generating response...")
    for chunk in model.generate_text_stream(prompt, params=generate_params):
        print(chunk, end="", flush=True)


def run_custom_prompt():
    client = APIClient(creds)
    project_id = os.getenv("PROJECT_ID")
    client.set.default_project(project_id)  # …or default_space("SPACE_ID")

    # 2.  Generation parameters (includes prompt-template variables)
    gen_params = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 1024,
        GenParams.STOP_SEQUENCES: [],  # ⚠️ turn off the default “\n\n”
        GenParams.TEMPERATURE: 0,
        GenParams.PROMPT_VARIABLES: {  # <— names must match template
            "context_year1": "passage: 3.2 Definition Unfall\nUnfall ist die plötzliche, nicht beabsichtigte schädigende Einwirkung eines ungewöhnlichen äusseren Faktors auf den menschlichen Körper, die eine Beeinträchtigung der körperlichen, geistigen oder psychischen Gesundheit oder den Tod zur Folge hat.",
            "context_year2": "passage: 3.2 Definition Unfall\n- a) Gesundheitsschädigung durch unfreiwilliges Einatmen von plötzlich ausströmenden Gasen oder Dämpfen;\n- b) Vergiftung oder Verletzung durch unabsichtliches Einnehmen von giftigen oder ätzenden Stoffen;\n- c) Unfreiwilliges Ertrinken, Erfrieren oder Hitzschlag;\n- d) Vom Willen des Versicherten unabhängige Verbrennungen, Verbrühungen, Einwirkungen von Blitzschlag oder elektrischem Strom.",
            "year_1": "",
            "year_2": "",
            "product_name": "Absicherungsplan",
            "user_query": "Wie wird ein Unfall definiert?",
            "file_name_1": "",
            "file_name_2": "",
        },
    }

    # 3.  Bind to **your** deployment
    deployed_model = ModelInference(
        deployment_id="b90940f5-9e78-4e90-ac61-46448ce376cf",
        params=gen_params,
        api_client=client,  # reuse the authenticated client
    )

    # 4.  Stream!
    print("Generating response...")
    for token in deployed_model.generate_text_stream():  # prompt=None for templates
        print(token, end="", flush=True)


if __name__ == "__main__":
    run_custom_prompt()
