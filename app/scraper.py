import os
import json
import re

import requests
from pathlib import Path
from dotenv import load_dotenv

from crawl4ai import *
import asyncio

load_dotenv(".env.llm")
BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"


def get_response(url, timeout=10):
    """
    Fetch a web page from the given URL
    Returns:
        requests.Response: The response object containing the page content
    """
    if not url.startswith("https://"):
        url = "https://" + url

    try:
        page = requests.get(url, timeout=timeout)
        page.raise_for_status()

        return page
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def download_json(url, output_filename):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, output_filename)

    print(f"Attempting to download from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"File successfully saved to: {save_path}")

    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")
    except IOError as e:
        print(f"Error writing file to disk: {e}")


def read_jina(url):
    url = "https://r.jina.ai/" + url
    response = get_response(url, timeout=1000)
    # print(response.text)

    text_content = {}
    if response and response.status_code == 200:
        lines = response.text.splitlines()

    # get abstract
    offset = 3
    marker = "Abstract"
    target_index = None

    for index, line in enumerate(lines):
        if marker in line:
            target_index = index + offset

            if target_index < len(lines):
                text_content[marker.lower()] = lines[target_index].strip()
            else:
                print(f"Found '{marker}' but offset {offset} is out of bounds.")
                return None
            break

    # get reviews
    lines = lines[target_index:]
    n = 1
    offset = 2
    marker = f"### Review #{n}"
    marker_next = f"### Review #{n+1}"
    marker_break = "Author Feedback"
    target_start = None
    target_end = None

    for index, line in enumerate(lines):
        if marker_break in line:
            print(marker, target_start, index)
            if target_start is not None and index is not None:
                text_content[f"review_{n}"] = "\n".join(
                    lines[target_start:index]
                ).strip()
            target_start = index + offset
            break
        if marker in line:
            target_start = index + offset

        if marker_next in line:
            target_end = index
            if target_start is not None and target_end is not None:
                text_content[f"review_{n}"] = "\n".join(
                    lines[target_start:target_end]
                ).strip()
                n += 1
                marker = f"### Review #{n}"
                marker_next = f"### Review #{n+1}"
                target_start = target_end
                target_end = None
            # print(text_content)

    print(text_content)


def change_json(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, filename)
    base_url = "https://papers.miccai.org"
    no_url = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # get only the first entry
        if data:
            data = [data[0]]
        for entry in data:
            if "url" in entry:
                entry["url"] = base_url + entry.get("url")
                text_content = read_jina(entry["url"])
                for key, value in text_content.items():
                    entry[key] = value
            else:
                no_url += 1

        if no_url > 0:
            print(f"Number of entries without URL: {no_url}")

        # with open(file_path, "w", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"File successfully updated: {file_path}")

    except IOError as e:
        print(f"Error reading or writing file: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")


text_content = {}


def extract_sections(markdown_text):
    """
    Automatically extract all H1 sections from markdown text.

    Returns:
        dict: Dictionary with section names as keys and content as values
    """
    sections = {}
    # Find all H1 headers
    h1_pattern = r"^# (.+)$"
    matches = list(re.finditer(h1_pattern, markdown_text, re.MULTILINE))

    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start_pos = match.end()

        # Determine end position (next H1 or end of text)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(markdown_text)

        content = markdown_text[start_pos:end_pos].strip()
        key = section_name.lower().replace(" ", "_")
        sections[key] = content

    return sections


def get_schema(html_sample=None, model=None):

    if model is not None:
        schema_file = f"{MEDIA_DIR}/home_schema_{model}.json"
    else:
        schema_file = f"{MEDIA_DIR}/home_schema.json"

    try:
        with open(schema_file, "r") as f:
            schema = json.load(f)
            print(f"Loaded Schema from {schema_file}")

    except FileNotFoundError:
        schema = JsonCssExtractionStrategy.generate_schema(
            html=html_sample,
            llm_config=LLMConfig(
                provider="gemini/gemini-2.5-pro",
                api_token=os.getenv("GEMINI_API_KEY"),
            ),
            query="From https://papers.miccai.org/miccai-2025/, i shared a sample html structure of a paper listing. Please generate a schema for this div extracting only the paper title, authors list, and the link to the paper information and reviews",
        )
        print(f"Generated Schema with GEMINI")

        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

    return schema


async def home_crawling(url, schema, model=None):
    try:
        with open(f"{MEDIA_DIR}/home_claude.json", "r") as f:
            data = json.load(f)
            print(f"Loaded Paper list")
            return data
    except FileNotFoundError:
        extraction_strategy = JsonCssExtractionStrategy(schema)
        config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

        async with AsyncWebCrawler() as crawler:
            result: CrawlResult = await crawler.arun(
                url=url,
                config=config,
            )

            if model is not None:
                file_name = f"{MEDIA_DIR}/home_{model}.json"
            else:
                file_name = f"{MEDIA_DIR}/home.json"

            if result.success:
                data = json.loads(result.extracted_content)
                with open(file_name, "w") as f:
                    json.dump(data, f, indent=2)
                return data
            else:
                raise Exception(f"Crawling failed for {url}")


async def paper_crawling(url, schema):
    extraction_strategy = JsonCssExtractionStrategy(schema)
    config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

    async with AsyncWebCrawler() as crawler:
        result: CrawlResult = await crawler.arun(
            url=url,
            config=config,
        )

        paper_id = url.split("/")[-1].replace(".html", "")
        file_name = f"{MEDIA_DIR}/{paper_id}.json"

        paper_md = result.markdown
        return extract_sections(paper_md)


def clean_section(paper, section, section_name):
    """
    # take only the site (between these brackets <> and put it back as a list) from: CT-RATE-Chinese dataset: <https://huggingface.co/datasets/SiyouLi/CT-RATE-Chinese> CT-RATE-Mini dataset: <https://huggingface.co/datasets/SiyouLi/CT-RATE-Mini> to [
    # {
    #     "CT-RATE-Chinese_dataset": "https://huggingface.co/datasets/SiyouLi/CT-RATE-Chinese"
    # },
    # {"CT-RATE-Mini_dataset": "https://huggingface.co/datasets/SiyouLi/CT-RATE-Mini"},]
    """

    if section_name == "link_to_the_dataset(s)":
        dataset_pattern = r"([^:]+):\s*<([^>]+)>"
        matches = re.findall(dataset_pattern, section)
        datasets = []
        for name, url in matches:
            datasets.append({name.strip().replace(" ", "_"): url.strip()})
        paper["Datasets"] = datasets

    elif section_name == "link_to_the_code_repository":
        code_pattern = r"<([^>]+)>"
        match = re.search(code_pattern, section)
        if match:
            paper["code"] = match.group(1).strip()
        else:
            paper["code"] = None

    elif section_name == "links_to_paper_and_supplementary_materials":
        # Extract DOI value (everything after "SpringerLink (DOI):" until newline)
        doi_pattern = r"SpringerLink \(DOI\):\s*(.+?)(?:\n|$)"
        doi_match = re.search(doi_pattern, section)

        # Extract Supplementary Material value (everything after "Supplementary Material:" until newline or end)
        supp_pattern = r"Supplementary Material:\s*(.+?)(?:\n|$)"
        supp_match = re.search(supp_pattern, section)

        if doi_match:
            doi_value = doi_match.group(1).strip()
            if doi_value.lower() in ["not yet available", "not available"]:
                paper["doi"] = None
            else:
                paper["doi"] = doi_value
        else:
            paper["doi"] = None

        if supp_match:
            supp_value = supp_match.group(1).strip()
            if supp_value.lower() in ["not submitted", "not available"]:
                paper["supp_material"] = None
            else:
                paper["supp_material"] = supp_value
        else:
            paper["supp_material"] = None
    else:
        paper[section_name] = section


async def get_data(base_url, url, model=None, html_sample=None):

    # Generate schema if not provided
    schema = get_schema(html_sample, model)

    # Home crawling
    home_data = await home_crawling(url, schema, model)

    # Paper crawling
    papers_list = []
    for paper in home_data:

        if paper["paper_link"]:
            if not paper.get("paper_link").startswith("https://"):
                paper["paper_link"] = base_url + paper.get("paper_link")
            paper_sections = await paper_crawling(paper["paper_link"], schema)
            valid_sections = [
                "abstract",
                "links_to_paper_and_supplementary_materials",
                "link_to_the_code_repository",
                "link_to_the_dataset(s)",
                "reviews",
                "author_feedback",
                "meta-review",
            ]
            # prendere solo le sezioni in sections e aggiungerle agli altri dati
            for paper_section in paper_sections:
                if paper_section in valid_sections:
                    clean_section(paper, paper_sections[paper_section], paper_section)

            papers_list.append(paper)

    # Write all papers as a list of dictionaries
    with open(f"{MEDIA_DIR}/papers_info.json", "w", encoding="utf-8") as f:
        json.dump(papers_list, f, ensure_ascii=False, indent=4)


def extract_specific(markdown_text, start, end=None):
    """
    Extract text between H1 headers with flexible pattern matching.

    Args:
        markdown_text: The markdown content to parse
        start: Pattern to match in the starting H1 header (case-insensitive)
        end: Optional pattern to match in the ending H1 header

    Returns:
        str: The extracted text between the headers
    """
    # Use case-insensitive matching for more flexibility
    if end:
        # Match text between two H1 headers
        pattern = rf"^# .*?{re.escape(start)}.*?$\n(.*?)(?=^# |\Z)"
    else:
        # Match from H1 to end of file
        pattern = rf"^# .*?{re.escape(start)}.*?$\n(.*?)(?=^# |\Z)"

    match = re.search(pattern, markdown_text, re.MULTILINE | re.DOTALL | re.IGNORECASE)

    if match:
        content = match.group(1).strip()
        # If end pattern is provided, check if we need to trim content
        if end and content:
            # Find where the next H1 with end pattern starts
            end_match = re.search(
                rf"(.*?)(?=^# .*?{re.escape(end)}|\Z)",
                content,
                re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )
            if end_match:
                content = end_match.group(1).strip()

        text_content[start.lower().replace(" ", "_")] = content
        return content

    text_content[start.lower().replace(" ", "_")] = ""
    return ""


if __name__ == "__main__":

    url = "https://papers.miccai.org/miccai-2025"

    asyncio.run(get_data("https://papers.miccai.org", url, model="claude"))
    # print("\nAll sections:", all_sections)
