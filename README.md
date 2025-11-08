
## Overview:

The project implements a chatbot capable of reading and extracting useful information from provided documents.

## Approach

### 1. Document Loading

The chatbot uses `pdfplumber` library to read PDF files.

### 2. Data Extraction

**Text Extraction**: The text context of each page is extracted and split using `CharacterTextSplitter` class.

**Table Extraction**: For table extraction i convert each table rows into string format for storage and easy retrieval.

**Image Extraction**: To extract data from image, we will suggest amazon textract or open source OCR tool.

### 3. Data Storage

The extracted text chunks and table data are stored in faiss vector database, so that is help in semantic retrieval.

### 4. Answer generation

When user submit the query the chatbot first convert query into embedding and then more related vectors are retrieve from vector database.

In next step query and chunks are send to LLm model for answer generation.

## Installation

To run code please create venv and install dependencies in it.
