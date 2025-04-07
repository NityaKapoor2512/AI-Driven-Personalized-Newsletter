import feedparser  # For parsing RSS feeds
import streamlit as st # For creating a web app UI
import google.generativeai as palm # For AI-based text summarization
from collections import defaultdict # For managing categorized data
import spacy # NLP library for text processing
from collections import defaultdict
import nltk # Natural Language Toolkit for sentiment analysis
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER sentiment analysis lexicon (needed for sentiment scoring)
nltk.download("vader_lexicon")

# Initialize the VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Configure the AI model with an API key (this key should be stored securely in real applications)
palm.configure(api_key="enter you key here")

# Define user profiles with their interests and preferred news sources
user_personas = {
    "Alex Parker": {
        "interests": ["AI", "cybersecurity", "blockchain", "startups", "programming"],
        "sources": ["TechCrunch", "Wired", "MIT Technology Review"]
    },
    "Priya Sharma": {
        "interests": ["Global markets", "startups", "fintech", "cryptocurrency", "economics"],
        "sources": ["Bloomberg", "Financial Times", "CoinDesk"]
    },
    "Marco Rossi": {
        "interests": ["Football", "F1", "NBA", "Olympic sports", "esports"],
        "sources": ["ESPN", "BBC Sport", "Sky Sports F1"]
    },
    "Lisa Thompson": {
        "interests": ["Movies", "celebrity news", "TV shows", "music", "books"],
        "sources": ["Variety", "Hollywood Reporter", "Billboard"]
    },
    "David Martinez": {
        "interests": ["Space exploration", "AI", "biotech", "physics", "renewable energy"],
        "sources": ["NASA", "Science Daily", "Ars Technica Science"]
    }
}

# Define RSS feed URLs for various news sources
rss_feeds = {
    # General News
    "BBC World News": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "The New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
        
    # Technology
    "TechCrunch": "https://techcrunch.com/feed/",
    "Wired": "https://www.wired.com/feed/rss",
    "MIT Technology Review": "https://www.technologyreview.com/feed/",
    
    # Finance & Business
    "Bloomberg": "https://www.bloomberg.com/feed/rss",
    "Financial Times": "https://www.ft.com/rss/home",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",

    
    # Sports
    "ESPN": "https://www.espn.com/espn/rss/news",
    "BBC Sport": "http://feeds.bbci.co.uk/sport/rss.xml",
    "Sky Sports F1": "https://www.skysports.com/rss/12040",
    
    # Entertainment
    "Variety": "https://variety.com/feed/",
    "Hollywood Reporter": "https://www.hollywoodreporter.com/feed/",
    "Billboard": "https://www.billboard.com/feed/",
    
    # Science & Space
    "NASA": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "Science Daily": "https://www.sciencedaily.com/rss/all.xml",
    "Ars Technica Science": "https://feeds.arstechnica.com/arstechnica/science",
}

# Function to summarize news articles using Google's AI model
def ai_summarize(text):
    try:
        response = palm.generate_text(
            model="models/text-bison-001",
            prompt=f"Summarize this news article in two sentences:\n{text}",
            temperature=0.5,
            max_output_tokens=100
        )
        return response.result if response.result else text # Return AI summary or original text if API fails
    except Exception:
        return text  # In case of an error, return the original text

# Function to analyze the sentiment of a news article using VADER
def analyze_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral"

# Load SpaCy's NLP model for text processing
nlp = spacy.load("en_core_web_sm")

# Function to extract important keywords from a given text
def extract_keywords(text):
    if not text or len(text) < 1:
        return set()
    
    doc = nlp(text)
    keywords = {token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and len(token.text) > 1}
    return keywords

# Function to fetch and process articles from an RSS feed
def fetch_articles(feed_url):
    feed = feedparser.parse(feed_url)
    articles = []
     # Loop through the first 10 articles from the feed
    for entry in feed.entries[:10]:  
        summary = ai_summarize(entry.summary)  # AI Summary
        sentiment = analyze_sentiment(summary)  # Sentiment Analysis
        articles.append({
            "title": entry.title,
            "summary": summary,
            "sentiment": sentiment,  # Store Sentiment Result
            "link": entry.link
        })
    return articles

# Function: Match Articles with User Interests using TF-IDF
def find_relevant_articles(user_interests, all_articles):
    vectorizer = TfidfVectorizer()
    
    # Collect keywords for all interests
    all_interest_keywords = set()
    for interest in user_interests:
        all_interest_keywords.update(extract_keywords(interest))

    # Convert to text for vectorization
    all_texts = [" ".join(all_interest_keywords)] + [article["title"] + " " + article["summary"] for article in all_articles]

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    user_vector = tfidf_matrix[0]  # First entry is user interests
    article_vectors = tfidf_matrix[1:]  # Remaining are articles

    # Compute relevance scores
    scores = (article_vectors * user_vector.T).toarray().flatten()
    
    # Sort and filter relevant articles
    sorted_articles = [all_articles[i] for i in scores.argsort()[::-1] if scores[i] > 0.1]
    
    return sorted_articles

# Function to curate personalized news based on user interests
def curate_news(user_name, user_info):
    personalized_news = defaultdict(list)

     # Fetch articles from the user's preferred sources
    for source in user_info["sources"]:
        if source in rss_feeds:
            articles = fetch_articles(rss_feeds[source])
            for article in articles:
                article_text = f"{article['title']} {article['summary']}"
                article_keywords = extract_keywords(article_text)

                 # Match articles with the user's interests
                for interest in user_info["interests"]:
                    if article_keywords.intersection(extract_keywords(interest)):
                        personalized_news[interest].append(article)

    return personalized_news 

# Function to generate a markdown newsletter file for the user
def generate_markdown_newsletter(user_name, curated_news):
    md_content = f"# {user_name}'s Personalized Newsletter\n\n"
     # Format the newsletter by category
    for category, articles in curated_news.items():
        md_content += f"## {category}\n\n"
        for article in articles:
            md_content += f"**[{article['title']}]({article['link']})** ({article['sentiment']})\n\n"
            md_content += f"{article['summary']}\n\n---\n\n"
    
     # Save the newsletter as a markdown file
    file_name = f"{user_name}_newsletter.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return file_name

# Streamlit UI Setup
st.title(" AI-Driven Personalized Newsletter")

# Dropdown for user selection
selected_user = st.selectbox("Select a User", list(user_personas.keys()))

# Button to generate the newsletter
if st.button("Generate Newsletter"):
    curated_news = curate_news(selected_user, user_personas[selected_user])
    st.subheader(f" {selected_user}'s Personalized Newsletter")

     # Display curated news in an expandable format
    for category, articles in curated_news.items():
        with st.expander(f" {category} ({len(articles)} articles)"):
            for article in articles:
                st.markdown(f"### [{article['title']}]({article['link']})")
                st.markdown(f"**Sentiment:** {article['sentiment']}")
                st.write(article['summary'])
                st.markdown("---")

    # Markdown Download
    markdown_file = generate_markdown_newsletter(selected_user, curated_news)
    with open(markdown_file, "rb") as file:
        st.download_button(" Download Newsletter", file, file_name=markdown_file, mime="text/markdown")
