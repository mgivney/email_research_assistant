#!/usr/bin/env python3
"""
Email script for generating and sending AI research summaries.

This script searches for AI-related content, summarizes it, and sends a daily email digest.
"""

from __future__ import print_function
import json
import os
import pathlib
import re
from typing import List, Dict, Any, Literal, Annotated

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import re
import unicodedata


# Configuration
SEARCH_TERMS = [
    "Agentic AI healthcare revenue cycle",
    "Autonomous healthcare Revenue Cycle",
    "Anthropic healthcare life-science",
    "Anthropic LinkedIn",
    "Anthropic healthcare LinkedIn"
]

required_environment_variables = [
    "SERPER_API_KEY",
    "SCRAPING_API_KEY",
    "RESEND_API_KEY",
    "OPENAI_API_KEY"
]

def validate_environment_variables():
    """
    Validate that all required environment variables are set.

    Checks each variable in the required_environment_variables list and raises
    a ValueError if any are missing. This ensures the script has access to all
    necessary API keys before attempting to run.

    Raises:
        ValueError: If any required environment variable is not set.
    """
    for var in required_environment_variables:
        if os.getenv(var) is None:
            raise ValueError(f"Environment variable {var} is not set")

class ResultRelevance(BaseModel):
    """Model for storing relevance check results."""
    explanation: str
    id: str


class RelevanceCheckOutput(BaseModel):
    """Model for storing all relevant results."""
    relevant_results: List[ResultRelevance]


class State(TypedDict):
    """State management for the LangGraph workflow."""
    messages: Annotated[list, add_messages]
    summaries: List[dict]
    approved: bool
    created_summaries: Annotated[List[dict], Field(description="The summaries created by the summariser")]
    email_template: str


class SummariserOutput(BaseModel):
    """Output format for the summarizer."""
    email_summary: str = Field(description="The summary email of the content")
    message: str = Field(description="A message to the reviewer requesting feedback")


class ReviewerOutput(BaseModel):
    """Output format for the reviewer."""
    approved: bool = Field(description="Whether the summary is approved")
    message: str = Field(description="Feedback message from the reviewer")


def search_serper(search_query: str) -> List[Dict[str, Any]]:
    """
    Search Google using the Serper API.
    
    Args:
        search_query: The search term to query
        
    Returns:
        List of search results with title, link, snippet, etc.
    """

    params = {
        "engine": "google",
        "q": "Agentic+AI+healthcare+revenue+cycle",
        "google_domain": "google.com",
        "hl": "en",
        "gl": "us",
        "api_key": os.getenv('SERPER_API_KEY') #"4a1706e9ad2e896b124b65f03231db5a9e4e71bc55ef2e9f54f4416a6e5a8340"
    }

    response = requests.get("https://serpapi.com/search", params=params)
    results = response.json()
    if 'organic_results' not in results:
        raise ValueError(f"No organic results found in results {results} for search query {search_query}")

    results_list = results['organic_results']

    return [
        {
            'title': result['title'],
            'link': result['link'],
            'snippet': result['snippet'],
            'search_term': search_query,
            'id': idx
        }
        for idx, result in enumerate(results_list[:20], 1)
    ]



def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Reads a markdown file containing a prompt template used for LLM interactions.
    Prompts are stored in the 'prompts/' directory with .md extension.

    Args:
        prompt_name: The name of the prompt file (without .md extension).

    Returns:
        The contents of the prompt file as a string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    with open(f"prompts/{prompt_name}.md", "r") as file:
        return file.read()


def check_search_relevance(search_results: Dict[str, Any]) -> RelevanceCheckOutput:
    """
    Analyze search results and determine the most relevant ones.
    
    Args:
        search_results: Dictionary containing search results to analyze
        
    Returns:
        RelevanceCheckOutput containing the most relevant results and explanation
    """
    prompt = load_prompt("relevance_check")
    prompt_template = ChatPromptTemplate.from_messages([("system", prompt)])
    llm = ChatOpenAI(model="gpt-4o").with_structured_output(RelevanceCheckOutput)
    
    return (prompt_template | llm).invoke({'input_search_results': search_results})


def convert_html_to_markdown(html_content: str) -> str:
    """
    Convert HTML content to markdown format.

    Parses HTML and converts common elements to their markdown equivalents:
    - Headers (h1-h6) to # syntax
    - Links to [text](url) format
    - Bold/strong to **text**
    - Italic/em to *text*
    - Unordered lists to - items
    - Ordered lists to numbered items

    Args:
        html_content: Raw HTML string to convert.

    Returns:
        Cleaned markdown string with normalized whitespace.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Convert headers
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(h.name[1])
        h.replace_with('#' * level + ' ' + h.get_text() + '\n\n')
    
    # Convert links
    for a in soup.find_all('a'):
        href = a.get('href', '')
        text = a.get_text()
        if href and text:
            a.replace_with(f'[{text}]({href})')
    
    # Convert formatting
    for tag, marker in [
        (['b', 'strong'], '**'),
        (['i', 'em'], '*')
    ]:
        for element in soup.find_all(tag):
            element.replace_with(f'{marker}{element.get_text()}{marker}')
    
    # Convert lists
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            li.replace_with(f'- {li.get_text()}\n')
    
    for ol in soup.find_all('ol'):
        for i, li in enumerate(ol.find_all('li'), 1):
            li.replace_with(f'{i}. {li.get_text()}\n')
    
    # Clean up text
    text = soup.get_text()
    return re.sub(r'\n\s*\n', '\n\n', text).strip()


def scrape_and_save_markdown(relevant_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scrape HTML content from URLs and save as markdown.
    
    Args:
        relevant_results: List of dictionaries containing search results
        
    Returns:
        List of dictionaries containing markdown content and metadata
    """
    pathlib.Path("scraped_markdown").mkdir(exist_ok=True)
    markdown_contents = []

    for result in relevant_results:
        if 'link' not in result:
            continue

        payload = {
            "api_key": os.getenv("SCRAPING_API_KEY"),
            "url": result['link'],
            "render_js": "true"
        }

        response = requests.get("https://scraping.narf.ai/api/v1/", params=payload)
        if response.status_code != 200:
            print(f"Failed to fetch {result['link']}: Status code {response.status_code}")
            continue

        filename = f"{result.get('id', hash(result['link']))}.md"
        filepath = os.path.join("scraped_markdown", filename)
        
        markdown_content = convert_html_to_markdown(response.content.decode())
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        markdown_contents.append({
            'url': result['link'],
            'filepath': filepath,
            'markdown': markdown_content,
            'title': result.get('title', ''),
            'id': result.get('id', '')
        })

    print(f"Successfully downloaded and saved {len(markdown_contents)} pages as markdown")
    return markdown_contents


def generate_summaries(markdown_contents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Generate summaries for markdown content using gpt-4o.
    
    Args:
        markdown_contents: List of dictionaries containing markdown content
        
    Returns:
        List of dictionaries containing summaries and URLs
    """
    pathlib.Path("markdown_summaries").mkdir(exist_ok=True)
    summary_prompt = load_prompt("summarise_markdown_page")
    summary_template = ChatPromptTemplate.from_messages([("system", summary_prompt)])
    llm = ChatOpenAI(model="gpt-4o")
    summary_chain = summary_template | llm
    
    summaries = []
    for content in markdown_contents:
        try:
            summary = summary_chain.invoke({
                'markdown_input': ' '.join(content['markdown'].split()[:2000])
            })
            
            summary_filename = f"summary_{content['id']}.md"
            summary_filepath = os.path.join("markdown_summaries", summary_filename)
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write(summary.content)
            
            summaries.append({
                'markdown_summary': summary.content,
                'url': content['url']
            })
                
        except Exception as e:
            print(f"Failed to summarize {content['filepath']}: {str(e)}")

    print(f"Successfully generated {len(summaries)} summaries")
    return summaries


def summariser(state: State) -> Dict:
    """
    Generate an email summary from the current workflow state.

    This is a LangGraph node function that takes the accumulated summaries
    and generates a formatted email digest using the configured LLM. It
    combines individual article summaries into a cohesive email following
    the provided template.

    Args:
        state: The current LangGraph state containing messages, summaries,
            and the email template.

    Returns:
        Dict containing:
            - messages: List of AIMessages with the email summary and feedback request
            - created_summaries: List containing the generated email summary
    """
    summariser_output = llm_summariser.invoke({
        "messages": state["messages"],
        "list_of_summaries": state["summaries"],
        "input_template": state["email_template"]
    })
    new_messages = [
        AIMessage(content=summariser_output.email_summary),
        AIMessage(content=summariser_output.message)
    ]
    return {
        "messages": new_messages,
        "created_summaries": [summariser_output.email_summary]
    }


def reviewer(state: State) -> Dict:
    """
    Review the generated email summary and provide feedback.

    This is a LangGraph node function that evaluates the summariser's output.
    It swaps message roles (AI <-> Human) to simulate a conversation where
    the reviewer critiques the summariser's work. The reviewer can either
    approve the summary or request revisions.

    Args:
        state: The current LangGraph state containing the conversation history
            and generated summaries.

    Returns:
        Dict containing:
            - messages: List with HumanMessage containing reviewer feedback
            - approved: Boolean indicating if the summary meets quality standards
    """
    converted_messages = [
        HumanMessage(content=msg.content) if isinstance(msg, AIMessage)
        else AIMessage(content=msg.content) if isinstance(msg, HumanMessage)
        else msg
        for msg in state["messages"]
    ]
    
    state["messages"] = converted_messages
    reviewer_output = llm_reviewer.invoke({"messages": state["messages"]})
    
    return {
        "messages": [HumanMessage(content=reviewer_output.message)],
        "approved": reviewer_output.approved
    }


def conditional_edge(state: State) -> Literal["summariser", END]:
    """
    Determine the next workflow step based on reviewer approval.

    This is a LangGraph conditional edge function that routes the workflow
    based on whether the reviewer approved the summary. If approved, the
    workflow ends. If not approved, it loops back to the summariser for
    revision.

    Args:
        state: The current LangGraph state containing the approval status.

    Returns:
        END if the summary is approved, "summariser" if revisions are needed.
    """
    return END if state["approved"] else "summariser"


def scrub_html_for_json(html: str) -> str:
    """
    Scrubs HTML content to be safe for JSON serialization.
    """
    # Decode bytes if necessary
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")

    # Normalize unicode (NFC handles combining characters, etc.)
    html = unicodedata.normalize("NFC", html)

    # Remove null bytes
    html = html.replace("\x00", "")

    # Remove non-printable control characters EXCEPT safe whitespace
    # Keeps: \t (tab), \n (newline), \r (carriage return)
    html = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", html)

    # Replace invalid unicode surrogate characters
    html = html.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    return html


def send_email(email_content: str):
    """
        Uses ReSend API to send an email.
    """
    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {os.getenv('RESEND_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "from": "RCM AI Bot<onboarding@resend.dev>",
            "to": 'mgivney@gmail.com',
            "subject": "RCM AI Bot: Your Agent",
            "html": scrub_html_for_json(email_content)
        },
    )
    if not response.ok:
        print(f"Resend error: {response.status_code} - {response.text}")
    response.raise_for_status()
    return response.json()


def main():
    """
    Main execution flow for the email research assistant.

    Orchestrates the complete pipeline:
    1. Loads environment variables from .env and validates them
    2. Searches for AI-related content using configured search terms
    3. Filters results for relevance using LLM analysis
    4. Scrapes relevant pages and converts to markdown
    5. Generates individual summaries for each article
    6. Runs a LangGraph workflow with summariser/reviewer agents
       to produce a polished email digest
    7. Sends the final approved email via Resend

    The workflow uses a feedback loop where the reviewer can request
    revisions from the summariser until the email meets quality standards.
    """
    load_dotenv()
    validate_environment_variables()
    # Search and filter results
    relevant_results = []

    results = search_serper('Agentic+AI+healthcare+revenue+cycle')
    filtered_results = check_search_relevance(results)
    relevant_ids = [r.id for r in filtered_results.relevant_results]
    filtered_results = [r for r in results if str(r['id']) in relevant_ids]
    relevant_results.extend(filtered_results)

    # Process content
    markdown_contents = scrape_and_save_markdown(relevant_results)
    summaries = generate_summaries(markdown_contents)

    # Set up LLM workflow
    llm = ChatOpenAI(model="gpt-4o")
    
    with open("email_template.md", "r") as f:
        email_template = f.read()

    summariser_prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("summariser")),
        ("placeholder", "{messages}"),
    ])

    reviewer_prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("reviewer")),
        ("placeholder", "{messages}"),
    ])

    global llm_summariser, llm_reviewer
    llm_summariser = summariser_prompt | llm.with_structured_output(SummariserOutput)
    llm_reviewer = reviewer_prompt | llm.with_structured_output(ReviewerOutput)

    # Configure and run graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("summariser", summariser)
    graph_builder.add_node("reviewer", reviewer)
    graph_builder.add_edge(START, "summariser")
    graph_builder.add_edge("summariser", "reviewer")
    graph_builder.add_conditional_edges('reviewer', conditional_edge)

    graph = graph_builder.compile()
    output = graph.invoke({"summaries": summaries, "email_template": email_template})

    # Send final email
    send_email(output["created_summaries"][-1])


if __name__ == "__main__":
    main()
