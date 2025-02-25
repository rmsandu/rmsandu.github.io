from flask_frozen import Freezer
from src.app import app, get_blog_info

freezer = Freezer(app)

if __name__ == '__main__':
    freezer.freeze()