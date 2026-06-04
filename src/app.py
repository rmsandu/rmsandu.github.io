import os
from datetime import date, datetime

from flask import Flask, render_template
import markdown
import frontmatter
import yaml

from src.markdown_extensions import BulmaImageExtension

app = Flask(__name__)

PAGES_DIR = os.path.join(os.path.dirname(__file__), "templates/pages")
BLOG_DIR = os.path.join(os.path.dirname(__file__), "templates/pages/blog")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_yaml_data(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def list_blog_files():
    return sorted(
        os.path.join(BLOG_DIR, f) for f in os.listdir(BLOG_DIR) if f.endswith(".md")
    )


def normalize_tags(tags):
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    return list(tags)


def normalize_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def get_blog_info(file):
    with open(file, "r", encoding="utf-8") as f:
        md_data = frontmatter.load(f)

    title = md_data.get("title", "No Title")
    subtitle = md_data.get("subtitle", "No Subtitle")
    coverImg = md_data.get("cover-img", "")
    thumbnailImg = md_data.get("thumbnail-img", "")
    tags = normalize_tags(md_data.get("tags", []))
    content = markdown.markdown(
        md_data.content,
        extensions=[BulmaImageExtension(), "fenced_code", "codehilite", "tables"],
    )
    filename = os.path.splitext(os.path.basename(file))[0]
    post_date = normalize_date(md_data.get("date"))

    return {
        "title": title,
        "subtitle": subtitle,
        "date": post_date,
        "content": content,
        "coverImg": coverImg,
        "thumbnailImg": thumbnailImg,
        "tags": tags,
        "filename": filename,
    }


def list_all_blog_info():
    blog_posts = [get_blog_info(file) for file in list_blog_files()]
    blog_posts.sort(key=lambda x: x["date"] or date.min, reverse=True)
    return blog_posts


def filter_blog_posts_by_tag(tag):
    all_blog_posts = list_all_blog_info()
    return [post for post in all_blog_posts if tag in post["tags"]]


# Custom filter to format the date
@app.template_filter("format_date")
def format_date(value, format="%B %d, %Y"):
    if value is None:
        return "No Date"
    return value.strftime(format)


@app.context_processor
def inject_globals():
    return {"current_year": datetime.now().year}


@app.route("/")
def index():
    return render_template(
        "pages/home.html",
        home=load_yaml_data("home.yaml"),
        blogPosts=list_all_blog_info(),
    )


@app.route("/publications.html")
def publications():
    return render_template("pages/publications.html", data=load_yaml_data("publications.yaml"))


@app.route("/<page_name>.html")
def render_page(page_name):
    page_path = os.path.join(PAGES_DIR, f"{page_name}.html")
    if os.path.exists(page_path):
        return render_template(f"pages/{page_name}.html")
    else:
        return "Page not found", 404


@app.route("/blogList.html")
def blogList():
    return render_template("pages/blogList.html", blogPosts=list_all_blog_info())


@app.route("/search/blogList/<tag>.html")
def search_tag(tag):
    filtered_blog_posts = filter_blog_posts_by_tag(tag)
    return render_template(
        "pages/blogList.html", blogPosts=filtered_blog_posts, selectedTag=tag
    )


@app.route("/blog/<page_name>.html")
def render_blog_page(page_name):
    path = os.path.join(BLOG_DIR, page_name + ".md")

    if os.path.exists(path):
        blog = get_blog_info(os.path.join(BLOG_DIR, page_name + ".md"))
        return render_template(
            "blog.jinja",
            title=blog["title"],
            subtitle=blog["subtitle"],
            date=blog["date"],
            coverImg=blog["coverImg"],
            thumbnailImg=blog["thumbnailImg"],
            tags=blog["tags"],
            content=blog["content"],
            page_title=page_name.replace("-", " ").title(),
        )
    else:
        return "Page not found", 404


if __name__ == "__main__":
    app.run(debug=True)
