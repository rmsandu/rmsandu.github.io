from flask_frozen import Freezer
from src.app import app, get_blog_info

freezer = Freezer(app)

@freezer.register_generator
def blog():
    for post in get_blog_info():
        yield {'page_name': post['file'][:-5]}  # Remove the .html extension

if __name__ == '__main__':
    freezer.freeze()