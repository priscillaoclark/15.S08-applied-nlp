import wikipedia

def get_wikipedia_content(page_title):
    try:
        # Search for the page
        search_results = wikipedia.search(page_title)
        
        # Find the most relevant page (usually the first result)
        for result in search_results:
            if "3M Company" in result or "Minnesota Mining and Manufacturing Company" in result:
                page = wikipedia.page(result)
                break
        else:
            return f"Error: Couldn't find a page matching '{page_title}'"
        
        # Get the title and content
        title = page.title
        content = page.content
        
        return f"Title: {title}\n\nContent:\n{content[:1000]}..."  # Truncating for brevity
    
    except wikipedia.exceptions.DisambiguationError as e:
        return f"DisambiguationError: {e.options}"
    except wikipedia.exceptions.PageError:
        return f"Error: Page '{page_title}' does not exist."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Fetch content for 3M page
page_title = "3M Company"
result = get_wikipedia_content(page_title)
print(result)