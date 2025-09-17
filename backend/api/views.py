import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType

from transformers import AutoTokenizer, AutoModel
import torch

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import requests
from bs4 import BeautifulSoup
import json
import re
import numpy as np

from django.apps import AppConfig

# it clears the collection
def clear_collection_in_dataBase():
    """Clear HtmlChunk collection: try delete+recreate first, else delete objects one-by-one."""
    try:
        # fast: remove entire collection/class
        client.collections.delete("HtmlChunk")
        # recreate schema so insert_chunks can run
        init_schema()
        print("Cleared HtmlChunk by deleting collection and re-created schema.")
        return
    except Exception as e:
        print("Could not delete collection directly:", e)

    # fallback: delete objects one-by-one (best-effort)
    try:
        collection = client.collections.get("HtmlChunk")
        objs = collection.query.fetch_objects(limit=1000)
        deleted = 0
        for obj in objs.objects:
            oid = getattr(obj, "id", None) or getattr(obj, "uuid", None) or obj._id if hasattr(obj, "_id") else None
            if not oid:
                # try properties-based fallback skip
                continue
            try:
                # preferred client API
                client.data_object.delete(oid, class_name="HtmlChunk")
                deleted += 1
            except Exception:
                # some clients expose different paths — try collection.data if exists
                try:
                    collection.data.delete_object(oid)
                    deleted += 1
                except Exception:
                    pass
        print(f"Cleared {deleted} objects from HtmlChunk (fallback).")
    except Exception as e:
        print("Fallback clear_collection_in_dataBase failed :", e)




# ---------------- Weaviate client ----------------
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://bm7lxp9isruc6ufd64pmow.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_credentials=Auth.api_key("SWZJMFJNK2ltNmkzZEI3bV9XUUJpK280RjdyQWNxbVY0dGxXb3BJUUNnRGs0VVZtRXZ5Rm1LZnB3TE9VPV92MjAw"),
)

# ---------------- Schema init ----------------

def init_schema():
    print('initial schema start')
    existing = client.collections.list_all()  # it's already a list of collection names (strings)
    if "HtmlChunk" not in existing:
        print('checking the html chunk condition')
        client.collections.create(
            name="HtmlChunk",
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="url", data_type=DataType.TEXT),
                Property(name="element_type", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="snippet", data_type=DataType.TEXT),
                Property(name="clean_html", data_type=DataType.TEXT),
            ],
            vectorizer_config=None  # we provide embeddings
        )
        print("Created HtmlChunk collection in Weaviate")
    else:
        print("HtmlChunk collection already exists")


# ---------------- Embeddings ----------------
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
emb_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = emb_model(**inputs)
    vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return vec

# it give macthing percentage between two vectors
def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

# ---------------- Content Processing ----------------
def clean_text(text):
    """Clean and normalize text content"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

# removes extra spaces before and after the text.
# ignores text if it's less than 15 characters (too small to matter)
# ignores text if it contains navigation/menu elements
# ignores text if it contains mostly non-alphabetic characters
# ignores text if it contains common boilerplate
# returns True if the text is meaningful, False otherwise
def removing_spaces_avoding_low_content_data(element, text):
    """Check if element contains meaningful content"""
    """Removing spaces"""
    text = text.strip()
    
    # Too short
    if len(text) < 15:
        return False
    
    # Skip navigation/menu elements
    classes = ' '.join(element.get('class', [])).lower()
    element_id = (element.get('id') or '').lower()
    if re.search(r'(nav|menu|footer|header|sidebar|cookie|breadcrumb|social)', classes + ' ' + element_id):
        return False
    
    # Skip elements with mostly non-alphabetic characters
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < len(text) * 0.5:
        return False
    
    # Skip common boilerplate
    if re.search(r'^(home|menu|login|register|search|subscribe|follow|contact|about|privacy|terms)$', text.lower().strip()):
        return False
    
    return True


# This function goes through the parsed HTML (soup) and pulls out meaningful text blocks (headings, paragraphs, sections, etc.).
# It avoids duplicates, ignores junk, and splits very long texts into smaller chunks.
# The goal: produce clean, structured pieces of content ready for storage or search.
def extract_clean_chunks(soup):
    """Extract clean, meaningful content chunks from HTML"""
    chunks = []
    
    # Priority elements for content extraction
    content_selectors = [
        'article', 'main', '[role="main"]', 
        '.content', '.post-content', '.entry-content',
        'section', '.section'
    ]
    
    # Try to find main content area first
    main_content = None
    for selector in content_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break   
    
    # If no main content area found, use body
    if not main_content:
        main_content = soup.find('body') or soup
    
    # Extract different types of content elements
    content_elements = main_content.find_all([
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',  # Headings
        'p',  # Paragraphs
        'article', 'section',  # Semantic elements
        'div',  # Divs (but we'll filter these more strictly)
        'blockquote', 'li'  # Other content elements
    ])
    
    processed_texts = set()  # Avoid duplicates
    
    for element in content_elements:
        text_content = element.get_text(separator=' ', strip=True)
        text_content = clean_text(text_content)
        
        # Skip if not meaningful or already processed
        if not removing_spaces_avoding_low_content_data(element, text_content) or text_content in processed_texts:
            continue
        
        # Limit chunk size
        if len(text_content) > 800:
            # Split long content into sentences
            sentences = re.split(r'[.!?]+', text_content)
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) < 400:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunks.append(create_content_chunk(element, current_chunk.strip()))
                        processed_texts.add(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk.strip():
                chunks.append(create_content_chunk(element, current_chunk.strip()))
                processed_texts.add(current_chunk.strip())
        else:
            chunks.append(create_content_chunk(element, text_content))
            processed_texts.add(text_content)
    
    return chunks

# This function packages one piece of text from HTML into a structured format.
# It figures out the element type (heading, section, paragraph, etc.), makes a short title/snippet, and keeps a clean HTML version.
# The output is a dictionary ready to store in the database or send to the frontend.
def create_content_chunk(element, text_content):
    """Create a structured content chunk"""
    # Determine content type and title
    tag_name = element.name
    element_type = "content"
    title = ""
    
    if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        element_type = "heading"
        title = text_content[:100]  # Use heading text as title
    elif tag_name in ['article', 'section']:
        element_type = "section"
        # Try to find a heading within the section
        heading = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if heading:
            title = heading.get_text(strip=True)[:100]
    elif tag_name == 'blockquote':
        element_type = "quote"
    elif tag_name == 'li':
        element_type = "list_item"
    else:
        element_type = "paragraph"
    
    # Create snippet (first 150 characters)
    snippet = text_content[:150] + "..." if len(text_content) > 150 else text_content
    
    # Create clean HTML representation
    clean_html = f'<{tag_name}>{text_content}</{tag_name}>'
    
    return {
        'content': text_content,
        'element_type': element_type,
        'title': title if title else f"{element_type.title()} Content",
        'snippet': snippet,
        'clean_html': clean_html
    }



# This function fetches a webpage, cleans out unwanted tags (scripts, styles, headers, etc.), and then extracts meaningful text chunks.
# It solves the problem of getting only useful readable content from messy raw HTML.
def fetch_and_clean_html(url):
    """Fetch URL and extract clean content chunks"""
    try:
        print(f"Fetching URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'noscript', 'iframe', 'nav', 'header', 'footer']):
            # removing the elements
            element.decompose()
        
        # Extract clean chunks
        chunks = extract_clean_chunks(soup)
        
        print(f"Extracted {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return []

# This function handles storing chunks into Weaviate:
# Deduplication: Before inserting, it fetches already stored content for the given URL and skips duplicates.
# Vectorization: For each new chunk, it generates embeddings with get_embedding(content).
# Insertion: Pushes both text + metadata + embedding vector into the HtmlChunk collection.
# Return: Number of successfully inserted chunks.
def insert_chunks_into_database(url, chunks):
    if not chunks:
        return 0

    collection = client.collections.get("HtmlChunk")
    inserted = 0

    # --- get existing contents for this URL (best-effort)
    existing_texts = set()
    try:
        existing_res = collection.query.fetch_objects(
            limit=1000,
            return_properties=["content", "url"]
        )
        for obj in existing_res.objects:
            props = obj.properties or {}
            if props.get("url", "").strip() == url.strip():
                existing_texts.add(props.get("content", "").strip())
    except Exception as e:
        print("Warning: could not fetch existing objects for dedupe:", e)
        existing_texts = set()

    # --- insert only new ones
    for chunk in chunks:
        content = chunk.get("content", "").strip()
        if not content:
            continue
        if content in existing_texts:
            # already inserted previously for this URL — skip
            continue
        try:
            vec = get_embedding(content)
            # ensure vector is plain list (weaviate client expects list/iterable)
            if hasattr(vec, "tolist"):
                vec_payload = vec.tolist()
            else:
                vec_payload = list(vec)
            collection.data.insert(
                properties={
                    "content": content,
                    "url": url,
                    "element_type": chunk.get("element_type", ""),
                    "title": chunk.get("title", ""),
                    "snippet": chunk.get("snippet", ""),
                    "clean_html": chunk.get("clean_html", ""),
                },
                vector=vec_payload
            )
            inserted += 1
        except Exception as e:
            print(f"Insert error for chunk (skipped): {e}")

    return inserted


# This function searches for relevant chunks in the database:
# Vector search: Uses cosine similarity to find chunks that are similar to the query.
# Text search: If vector search doesn't find good matches, it does a plain text search.
# Return: List of chunks sorted by similarity score.
def search_chunks_in_database(query, top_k=10):
    """Search for relevant chunks"""
    try:
        print(f"Searching for query: '{query}'")
        q_vec = get_embedding(query)
        print(f"Query vector shape: {q_vec.shape}")
        
        collection = client.collections.get("HtmlChunk")
        
        # First, try to get some results to debug
        res = collection.query.near_vector(
            near_vector=q_vec.tolist(),
            limit=top_k,
            return_properties=["content", "url", "element_type", "title", "snippet", "clean_html"]
        )
        
        print(f"Weaviate returned {len(res.objects)} objects")
        
        results = []
        for i, obj in enumerate(res.objects):
            props = obj.properties
            content = props.get("content", "")
            print(f"Object {i}: content preview: '{content[:100]}...'")
            
            # Calculate similarity
            chunk_vec = get_embedding(content)
            sim = cosine_sim(q_vec, chunk_vec)
            match_pct = max(0, round(sim * 100, 2))
            
            print(f"Object {i}: similarity = {sim}, match_pct = {match_pct}")
            
            results.append({
                "title": props.get("title", ""),
                "snippet": props.get("snippet", ""),
                "content": content,
                "element_type": props.get("element_type", ""),
                "clean_html": props.get("clean_html", ""),
                "match_percentage": match_pct
            })
        
        # Sort by match percentage
        results.sort(key=lambda x: x['match_percentage'], reverse=True)
        print(f"Returning {len(results)} results")
        
        # If no good matches, let's also try a simple text search as fallback
        if not results or max([r['match_percentage'] for r in results]) < 10:
            print("Low similarity scores, trying text-based search...")
            text_results = []
            
            # Get all recent chunks and do text matching
            all_res = collection.query.fetch_objects(
                limit=50,
                return_properties=["content", "url", "element_type", "title", "snippet", "clean_html"]
            )
            
            query_lower = query.lower()
            for obj in all_res.objects:
                content = obj.properties.get("content", "")
                if query_lower in content.lower():
                    # Simple text match score
                    match_score = content.lower().count(query_lower) * 20
                    match_score = min(match_score, 100)  # Cap at 100%
                    
                    text_results.append({
                        "title": obj.properties.get("title", ""),
                        "snippet": obj.properties.get("snippet", ""),
                        "content": content,
                        "element_type": obj.properties.get("element_type", ""),
                        "clean_html": obj.properties.get("clean_html", ""),
                        "match_percentage": match_score
                    })
            
            text_results.sort(key=lambda x: x['match_percentage'], reverse=True)
            print(f"Text search found {len(text_results)} matches")
            
            # Return text results if they're better
            if text_results and (not results or text_results[0]['match_percentage'] > results[0]['match_percentage']):
                return text_results[:top_k]
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return []

# api hit's this funtion 
@csrf_exempt
def test_search(request):
    print("---- test_search called ----")
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"})

    try:
        data = json.loads(request.body)
        url = data.get("url", "").strip()
        query = data.get("query", "").strip()

        if not url or not query:
            return JsonResponse({"error": "URL and query are required"})
        
        clear_collection_in_dataBase()    # <--- clear previous data first

        print(f"Processing URL: {url}, Query: {query}")

        # Fetch and extract content (returns list of chunk dicts)
        chunks = fetch_and_clean_html(url)

        # this is the case where no meaningful content is found on the webpage
        # this is the case where the webpage is not found
        # return empty list
        if not chunks:
            return JsonResponse({
                "url": url,
                "query": query,
                "top_chunks": [],
                "message": "No meaningful content found on the webpage",
                "debug": {"chunks_extracted": 0}
            })

        # Insert chunks into vector DB (insert_chunks_into_database should return inserted count)
        inserted_count = insert_chunks_into_database(url, chunks)

        # Search for relevant chunks in the database
        results = search_chunks_in_database(query, top_k=10)  # expects list of dicts with 'content','clean_html','match_percentage',...

        # --- Dedupe identical content (preserve order) ---
        seen = set()
        unique_results = []
        for r in results:
            content_key = (r.get("content") or "").strip()
            if not content_key or content_key in seen:
                continue
            seen.add(content_key)
            unique_results.append(r)
        results = unique_results

        # Format for frontend (only keys frontend needs: chunk_text, chunk_html, match_percentage)
        formatted_results = []
        for r in results:
            match_pct = r.get("match_percentage", 0) or 0
            formatted_results.append({
                "chunk_text": r.get("content", ""),
                "chunk_html": r.get("clean_html", ""),
                "match_percentage": match_pct,
                # optional extras for demo
                "title": r.get("title", ""),
                "snippet": r.get("snippet", "")
            })

        print(f"Chunks extracted: {len(chunks)}, inserted: {inserted_count}, results returned: {len(formatted_results)}")

        return JsonResponse({
            "url": url,
            "query": query,
            "top_chunks": formatted_results,
            "debug": {
                "chunks_extracted": len(chunks),
                "chunks_inserted": inserted_count,
                "results_found": len(results),
                "results_returned": len(formatted_results)
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        print("test_search error:", e)
        import traceback; traceback.print_exc()
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)
 
