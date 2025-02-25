from markdown.extensions import Extension
from markdown.inlinepatterns import ImageInlineProcessor
import xml.etree.ElementTree as etree
from flask import url_for

class BulmaImageExtension(Extension):
    def extendMarkdown(self, md):
        IMAGE_LINK_RE = r'!\[([^\]]*)\]\(([^)]*)\)'  # Markdown image pattern
        md.inlinePatterns.register(BulmaImageProcessor(IMAGE_LINK_RE, md), 'image', 175)

class BulmaImageProcessor(ImageInlineProcessor):
    def handleMatch(self, m, data):
        el = etree.Element('img')
        el.set('src', m.group(2))
        el.set('alt', m.group(1))
        
        src = el.get('src')
        if src:
            el.set('src', url_for('static', filename=src))
            el.set('class', 'blog-image')
        
        figure = etree.Element('figure')
        figure.set('class', 'image')
        figure.append(el)
        
        return figure, m.start(0), m.end(0)

def makeExtension(**kwargs):
    return BulmaImageExtension(**kwargs)