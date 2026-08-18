import os
from datetime import date, datetime, time
from xml.sax.saxutils import escape

from flask import Flask, render_template, request, url_for, Response
import markdown
import frontmatter
import yaml

from src.markdown_extensions import BulmaImageExtension

app = Flask(__name__)

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
    return {
        "current_year": datetime.now().year,
        "canonical_url": request.url_root.rstrip("/") + request.path,
    }


@app.route("/")
def index():
    return render_template(
        "pages/home.html",
        home=load_yaml_data("home.yaml"),
        timeline=load_yaml_data("timeline.yaml").get("timeline", []),
        blogPosts=list_all_blog_info(),
    )


@app.route("/publications.html")
def publications():
    return render_template("pages/publications.html", data=load_yaml_data("publications.yaml"))


@app.route("/consulting.html")
def consulting():
    return render_template("pages/consulting.html")


@app.route("/blogList.html")
def blogList():
    return render_template("pages/blogList.html", blogPosts=list_all_blog_info())


@app.route("/search/blogList/<tag>.html")
def search_tag(tag):
    filtered_blog_posts = filter_blog_posts_by_tag(tag)
    return render_template(
        "pages/blogList.html", blogPosts=filtered_blog_posts, selectedTag=tag
    )


def citation_key(page_name, post_date):
    slug = page_name[11:].replace("-", "")  # strip the leading "YYYY-MM-DD-"
    year = post_date.year if post_date else ""
    return f"sandu{year}{slug}"


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
            citation_key=citation_key(page_name, blog["date"]),
        )
    else:
        return "Page not found", 404


@app.route("/sitemap.xml")
def sitemap():
    static_pages = ["index", "publications", "consulting", "blogList"]
    urls = [url_for(endpoint, _external=True) for endpoint in static_pages]
    urls += [
        url_for("render_blog_page", page_name=post["filename"], _external=True)
        for post in list_all_blog_info()
    ]
    xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    xml += [f"<url><loc>{url}</loc></url>" for url in urls]
    xml.append("</urlset>")
    return Response("\n".join(xml), mimetype="application/xml")


@app.route("/feed.xml")
def feed():
    posts = list_all_blog_info()
    items = []
    for post in posts:
        post_url = url_for("render_blog_page", page_name=post["filename"], _external=True)
        pub_date = datetime.combine(post["date"] or date.today(), time()).strftime("%a, %d %b %Y %H:%M:%S GMT")
        items.append(
            "<item>"
            f"<title>{escape(post['title'])}</title>"
            f"<link>{post_url}</link>"
            f"<guid>{post_url}</guid>"
            f"<pubDate>{pub_date}</pubDate>"
            f"<description>{escape(post['subtitle'])}</description>"
            "</item>"
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel>'
        "<title>Raluca-Maria Sandu — Blog</title>"
        f"<link>{url_for('blogList', _external=True)}</link>"
        "<description>Notes on applied machine learning, computer vision, and generative AI.</description>"
        "<language>en-us</language>"
        + "".join(items)
        + "</channel></rss>"
    )
    return Response(xml, mimetype="application/rss+xml")


@app.route("/robots.txt")
def robots():
    lines = ["User-agent: *", "Allow: /", f"Sitemap: {url_for('sitemap', _external=True)}"]
    return Response("\n".join(lines), mimetype="text/plain")


@app.route("/CNAME")
def cname():
    return Response("rmsandu.net", mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)
