# Data Management for AI - Research Project

This repository contains our research project for the Data Management for AI course. 

## Project Overview

Extracting structured information from unstructured text remains a significant challenge across various domains. For example, in healthcare, clinical notes, research reports, and patient records may contain valuable but inconsistently formatted data. Traditional rule-based approaches, such as regular expressions, are often brittle, requiring exact keyword matches and extensive hand-crafting to accommodate variations in language. A key limitation of these traditional methods is their inability to generalize across diverse expressions while maintaining high precision. To address this, we propose leveraging large language models (LLMs) for semantic pattern matching. Given a user-specified semantic regex containing multiple semantic symbols and an unstructured medical record, we explored two approaches: (1) using an LLM to directly predict whether the semantic regex matches the record, and (2) using an LLM to extract semantic symbols and using rule-based approaches like finite state machines and set logic to identify patterns. Our experimental results suggested that the first approach was both faster and more accurate. We evaluated the approaches across a variety of different models, among which we determined GPT-4o to be the best-performing. These results support the conclusion that in text extraction and classification tasks, it is best to delegate as much of the logic to the LLM because it is equipped to handle the varied nature of natural language.

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/natcasd/chatdbt.git
```
2. Set up your environment:
   - Copy `.env.example` to `.env`
   - Add your API keys to `.env`

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Sample code to run approaches is in the notebooks in src/.

## Team Members

By Nathan, Justin, Yandi, Noah

