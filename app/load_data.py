import json
import sys
from pathlib import Path
from django.db import transaction
import argparse
import requests

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import django
from django.core.files.base import ContentFile
import os
from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env.local")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
django.setup()

from webApp.models import Paper, Dataset, Conference


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Load papers from JSON file into the database"
    )
    parser.add_argument(
        "confName",
        type=str,
        help="Name of the conference",
    )
    parser.add_argument(
        "confYear",
        type=int,
        help="Year of the conference",
    )
    parser.add_argument(
        "confUrl",
        type=str,
        help="URL of the conference website",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="media/papers_info.json",
        help="Path to JSON file",
    )

    args = parser.parse_args()
    return args


def save_pdf(paper, conference_name, conference_year):
    """Download and save PDF file for a paper."""
    if not paper.pdf_url:
        print(f"No PDF URL for paper: {paper.title}...")
        return False

    try:
        print(f"  ↓ Downloading PDF from: {paper.pdf_url}")
        response = requests.get(paper.pdf_url, timeout=30)
        response.raise_for_status()
        # per ottenere i byte che compongono il PDF
        pdf_content = response.content

        # Verify it's actually a PDF
        if not pdf_content.startswith(b"%PDF"):
            print(f"Downloaded file is not a valid PDF")
            return False

    except requests.RequestException as e:
        print(f"Error during the download of {paper.pdf_url}: {e}")
        return
    name = paper.pdf_url.split("/")[-1]
    name = f"{conference_name}_{conference_year}_{name}"
    pdf_file = ContentFile(pdf_content, name=name)
    print(f"File saved: {pdf_file.name}")
    return pdf_file


def load_data(file_path, conference_name, conference_year, conference_url):
    """
    Load papers from JSON file into the database.

    Args:
        file_path: Path to the JSON file containing papers data
        conference_name: Name of the conference
        year: Year of the conference
        conference_url: URL of the conference
    """

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Loading {len(papers)} papers for {conference_name} {conference_year}...")
    print(f"Conference URL: {conference_url}")

    # Create or get conference
    conference, created = Conference.objects.get_or_create(
        name=conference_name,
        year=conference_year,
        url=conference_url,
    )
    if created:
        print(f"✓ Created conference: {conference_name} {conference_year}")
    else:
        conference.save()
        print(f"↻ Using existing conference: {conference_name} {conference_year}")

    created_papers = 0
    updated_papers = 0
    errors = 0

    for paper in papers:

        with transaction.atomic():

            # authors_list = []
            # if paper.get("authors"):
            #     authors = [name.strip() for name in paper["authors"].split(",")]
            #     for author in authors:
            #         if author:
            #             author, created = Author.objects.get_or_create(name=author)
            #             authors_list.append(author)
            #             if created:
            #                 print(f"  ✓ Created author: {author}")

            datasets_list = []
            if paper.get("datasets"):
                if isinstance(paper["datasets"], dict):
                    for name, url in paper["datasets"].items():
                        dataset, created = Dataset.objects.get_or_create(
                            name=name, url=url
                        )
                        datasets_list.append(dataset)
                    paper.pop("datasets")

            exsisting_paper = None
            if paper.get("paper_url"):
                exsisting_paper = Paper.objects.filter(
                    paper_url=paper["paper_url"]
                ).first()

            if not exsisting_paper and paper.get("doi"):
                exsisting_paper = Paper.objects.filter(doi=paper["doi"]).first()

            if exsisting_paper:
                # Update existing paper
                for key, value in paper.items():
                    setattr(exsisting_paper, key, value)
                exsisting_paper.save()
                updated_papers += 1
                print(f"↻ Updated paper: {exsisting_paper.title}...")
                paper = exsisting_paper
            else:
                # Create new paper
                paper = Paper.objects.create(**paper)
                created_papers += 1
                print(f"✓ Created paper: {paper.title}...")

            pdf_file = save_pdf(paper, conference_name, conference_year)
            paper.pdf_file.save(pdf_file.name, pdf_file, save=True)
            # Set many-to-many relationship
            # paper.authors.set(authors_list)
            conference.papers.add(paper)
            paper.datasets.set(datasets_list)

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Papers created: {created_papers}")
    print(f"  Papers updated: {updated_papers}")
    if errors > 0:
        print(f"  Errors: {errors}")
    print("=" * 50)


if __name__ == "__main__":

    args = parseArguments()

    if args is None:
        print(
            f"Information about the conference not provided to {__file__.split('/')[-1]}"
        )
        exit(1)

    json_file = args.file if args.file else BASE_DIR / "media" / "papers_info.json"

    print(f"Loading papers from: {json_file}")
    load_data(json_file, args.confName, args.confYear, args.confUrl)
