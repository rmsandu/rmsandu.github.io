from flask import Flask, render_template, request, url_for
import os
import markdown
import frontmatter
from datetime import datetime
from src.markdown_extensions import BulmaImageExtension
import yaml

app = Flask(__name__)

PAGES_DIR = os.path.join(os.path.dirname(__file__), 'templates/pages')
BLOG_DIR = os.path.join(os.path.dirname(__file__), 'templates/pages/blog')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def list_blog_files():
    return [os.path.join(BLOG_DIR, f) for f in os.listdir(BLOG_DIR) if f.endswith('.md')]

def get_blog_info(file):
    with open(file, "r", encoding="utf-8") as f:
            md_data = frontmatter.load(f)

    title = md_data.get('title', 'No Title')
    subtitle = md_data.get('subtitle', 'No Subtitle')
    coverImg = md_data.get('cover-img', '')
    thumbnailImg = md_data.get('thumbnail-img', '')
    tags = md_data.get('tags', '')
    content = markdown.markdown(md_data.content, extensions=[BulmaImageExtension(), 'fenced_code', 'codehilite'])
    filename = os.path.splitext(os.path.basename(file))[0]

    # Convert date string to datetime object
    try:
        date = md_data.get('date', 'No Date')
    except ValueError:
        date = None

    return {'title': title, 'subtitle': subtitle, 'date': date, 'content': content, 'coverImg': coverImg, 'thumbnailImg': thumbnailImg, 'tags': tags, 'filename': filename}
            

def list_all_blog_info():
    blog_posts = []

    for file in list_blog_files():       
        blog_posts.append(get_blog_info(file))

    blog_posts.sort(key=lambda x: x['date'], reverse=True)
    return blog_posts

def filter_blog_posts_by_tag(tag):
    all_blog_posts = list_all_blog_info()
    return [post for post in all_blog_posts if tag in post['tags']]

# Custom filter to format the date
@app.template_filter('format_date')
def format_date(value, format='%B %d, %Y'):
    if value is None:
        return 'No Date'
    return value.strftime(format)

@app.route('/')
def index():
    return render_template('pages/home.html', blogPosts=list_all_blog_info())

@app.route('/publications.html')
def publications():
    with open(os.path.join(DATA_DIR, 'publications.yaml'), 'r', encoding='utf-8') as file:
        return render_template('pages/publications.html', data = yaml.safe_load(file)) 

@app.route('/<page_name>.html')
def render_page(page_name):
    page_path = os.path.join(PAGES_DIR, f'{page_name}.html')
    if os.path.exists(page_path):
        return render_template(f'pages/{page_name}.html')
    else:
        return "Page not found", 404
    
@app.route('/blogList.html')
def blogList():
    return render_template('pages/blogList.html', blogPosts=list_all_blog_info())

@app.route('/search/blogList/<tag>.html')
def search_tag(tag):
    filtered_blog_posts = filter_blog_posts_by_tag(tag)
    return render_template('pages/blogList.html', blogPosts=filtered_blog_posts, selectedTag=tag)
    
@app.route('/blog/<page_name>.html')
def render_blog_page(page_name):
    path = os.path.join(BLOG_DIR, page_name + ".md")

    if os.path.exists(path):
        blog = get_blog_info(os.path.join(BLOG_DIR, page_name + ".md"))
        return render_template('blog.jinja',
                               title=blog['title'],
                               subtitle=blog['subtitle'],
                               date=blog['date'],
                               coverImg=blog['coverImg'],
                               thumbnailImg=blog['thumbnailImg'],
                               tags=blog['tags'],
                               content=blog['content'],
                               page_title=page_name.replace('-', ' ').title())
    else:
        return "Page not found", 404

if __name__ == '__main__':
    app.run(debug=True)