from openai import max_retries
from pypdf import PdfReader
from pathlib import Path
import sys
import os
from dotenv import load_dotenv
from google import genai
import argparse
import time

api_key = os.getenv("GEMINI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env.llm")
PDF_DIR = Path(__file__).resolve().parent / "media" / "pdf"


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Load papers from JSON file into the database"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to text file",
    )
    args = parser.parse_args()
    return args


def get_instruction(category):
    if category == "cleaning":
        return "### INSTRUCTION: You are an expert data cleaner. Follow these steps: 1. Analyse the DATA. 2. Clean the DATA by removing any noise related the fact the text provided comes from a pdf 3. Fix the synthax of all the formulas and math characters using a mathjax one 4. Do not provide any explanation or further text. Only provide the cleaned information. —- ### DATA:"
    elif category == "information":
        return '### INSTRUCTION: You are an expert at reading scientific papers. Follow these steps: 1. Analyse the PAPER. 2. Extract these information from the PAPER, the author mail "authors_mail" (as a dictionary) the amount of emails always correspond to the number of authors, the authors should be written in the output in the same order as you found in the paper, the datasets used in the paper "datasets" (as a dictionary), the code of the repository "code_url" 3. Structure your response using the following JSON schema: { "authors_mail":{"mail@example.com":"Name Surname"},"datasets":{"name_1":"url","name_2":"url",...}, "code_url": "..."} 4. If a value is not found put null 5. Do not provide any explanation or further text Only provide the JSON. —- ### PAPER:'


def read_pdf(pdf_name):
    reader = PdfReader(PDF_DIR / pdf_name)
    pages = reader.pages
    text = ""
    for page in pages:
        text = text + page.extract_text()
    return text


def main():
    text = read_pdf("miccai_2025_0308_paper.pdf")
    with open(PDF_DIR / "miccai_2025_0308_paper.txt", "w") as text_file:
        text_file.write(text)

    client = genai.Client(api_key=api_key)

    # ~~~~~~~~~ Set What Extract from the pdf ~~~~~~~~~#
    extraction_category = "information"  # cleaning / information
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Retry logic for server errors
    count = 0
    retry_delay = 1  # seconds

    while True:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=get_instruction(extraction_category) + text,
            )
            break  # Success, exit retry loop
        except Exception as e:

            print(
                f"Error: {e}. Retrying in {retry_delay} seconds... (Attempt {count + 1})"
            )
            time.sleep(retry_delay)
            if retry_delay < 30:
                retry_delay += 2  # Exponential backoff

    with open(
        PDF_DIR / f"miccai_2025_0308_paper_{extraction_category}.txt", "w"
    ) as text_file:
        text_file.write(response.text)


if __name__ == "__main__":
    main()
