import json
import os
import sys

from openai import OpenAI

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
from web.settings import BASE_DIR

# from webApp.models import Paper, Dataset, Conference
from dotenv import load_dotenv

sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env.llm")
PDF_DIR = BASE_DIR / "media" / "pdf"
moonshot_api_key = os.getenv("MOONSHOT_API_KEY")


# ================================= KIMI K2 ========================================


# ========== STANDARD USE ============
def kimi_standard(client: OpenAI, pdf_text: str, system_prompt: str, model: str):

    messages = [
        {
            "role": "system",
            "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI, excelling in English conversations.",
        },
        {
            "role": "system",
            "content": system_prompt,
        },  # <-- Submit the system prompt with the output format to Kimi
        {"role": "user", "content": pdf_text},
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        response_format={
            "type": "json_object"
        },  # <-- Use the response_format parameter to specify the output format as json_object
    )

    return completion.choices[0].message.content


# ============ UPLOAD FILES ============


def kimi_files(client: OpenAI, file: str, system_prompt: str, model: str):

    file_object = client.files.create(file=file, purpose="file-extract")

    # Retrieve the result
    file_content = client.files.content(file_id=file_object.id).text

    # Include it in the request
    messages = [
        {
            "role": "system",
            "content": file_content,
        },
        {
            "role": "user",
            "content": system_prompt,
        },
    ]

    # Then call chat-completion to get Kimi's response

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        response_format={"type": "json_object"},
    )

    return completion.choices[0].message


def main():

    client = OpenAI(
        api_key=moonshot_api_key,
        base_url="https://api.moonshot.ai/v1",
    )
    model = "kimi-k2-thinking"
    criterion = "Annotation: This criterion evaluates information on how annotations were performed, the expertise level of the annotators, which metrics were used, and how much the annotators agreed with one another. "
    json_format = """
    {
        "text": "Extracted text related to the criterion",
        "score": "Score based on criterion"
    }
    """
    system_prompt = f"""
    You are an intelligent scientific paper evaluator, responsible for analyzing scientific papers provided and giving them a different score based on given criterions. Your reply can be only text in JSON format as specified below.
    Based on the criterion provided, one or more, you have to find and put in the answer the text related to the CRITERION ("text field") and a integer score from 0 to 2 ("score" field) based on how well the paper addresses that CRITERION.
    CRITERION - {criterion}
    
    Please output your reply in the following JSON format:
    
    {json_format}
    
    Note: Please place the text information in the `text` field and the corresponding score in the `score` field.
        For the score you can put 0 if there is no information about the annotation process, 1 if there is some information but it is incomplete or unclear, and 2 if there is detailed and clear information about the annotation process, including the expertise of annotators, metrics used, and inter-annotator agreement.
    """
    # Check the amount of character in a string

    # response = kimi_files(client, PDF_DIR / "miccai_2025_0308_paper.pdf", system_prompt)

    with open(PDF_DIR / "miccai_2025_0308_paper.txt", "r") as f:
        pdf_text = f.read()
        input_tokens = len(system_prompt + pdf_text) / 3.2
        response = kimi_standard(client, pdf_text, system_prompt)

    # save the response (json file)
    with open(PDF_DIR / "miccai_2025_0308_response.json", "w") as f:
        json.dump(response, f, indent=2)

    output_tokens = len(response) / 3.2

    print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")


if __name__ == "__main__":
    main()
