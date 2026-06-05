from flask_frozen import Freezer

from src.app import app, list_all_blog_info

freezer = Freezer(app)


@freezer.register_generator
def render_page():
    yield {"page_name": "resume"}


@freezer.register_generator
def render_blog_page():
    for post in list_all_blog_info():
        yield {"page_name": post["filename"]}


@freezer.register_generator
def search_tag():
    tags = set()
    for post in list_all_blog_info():
        tags.update(post["tags"])

    for tag in sorted(tags):
        yield {"tag": tag}


if __name__ == '__main__':
    freezer.freeze()
