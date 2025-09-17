# Backend – Website Semantic Search

This is the Django backend for the Website Semantic Search project.  
It crawls and parses website content, stores embeddings in a vector database, and provides semantic search results through an API.

---

## Features
- Django REST API (`/api/search/`)
- Scrapes and tokenizes website content
- Stores vector embeddings in **Weaviate**
- Returns top matching chunks with:
  - Title
  - Snippet text
  - HTML content
  - Match percentage

---

## Tech Stack
- Django
- Django REST Framework
- Requests
- BeautifulSoup
- Weaviate Client

---

## Prerequisites
- Python 3.10+
- pip

---

## Setup & Run

```bash
# go inside backend folder
cd backend

# create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# install dependencies
pip install -r requirements.txt

# run migrations
python manage.py migrate

# start server
python manage.py runserver
