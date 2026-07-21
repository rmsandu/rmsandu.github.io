from datetime import date
from pathlib import Path

import pytest
from flask_frozen import Freezer

import src.app as app_module
from src.app import (
    app,
    filter_blog_posts_by_tag,
    format_date,
    get_blog_info,
    list_all_blog_info,
    list_blog_files,
    load_yaml_data,
)
from src.markdown_extensions import BulmaImageProcessor


def write_post(path, title, post_date, tags=None, body="Body text"):
    tag_text = tags or "[]"
    path.write_text(
        "\n".join(
            [
                "---",
                f"date: {post_date}",
                f"title: {title}",
                "subtitle: Test subtitle",
                "cover-img: img/test.png",
                "thumbnail-img: img/test-thumb.png",
                f"tags: {tag_text}",
                "---",
                "",
                body,
            ]
        ),
        encoding="utf-8",
    )


def test_list_blog_files_returns_only_markdown(tmp_path, monkeypatch):
    write_post(tmp_path / "b.md", "B", "2024-01-01")
    write_post(tmp_path / "a.md", "A", "2024-01-02")
    (tmp_path / "template.md.template").write_text("ignore", encoding="utf-8")

    monkeypatch.setattr(app_module, "BLOG_DIR", str(tmp_path))

    assert [Path(path).name for path in list_blog_files()] == ["a.md", "b.md"]


def test_get_blog_info_parses_frontmatter_and_markdown(tmp_path):
    post = tmp_path / "sample-post.md"
    write_post(
        post,
        "Sample Post",
        "2024-05-04",
        "[vision, ai]",
        "Intro\n\n![Alt text](img/example.png)",
    )

    with app.test_request_context():
        info = get_blog_info(str(post))

    assert info["title"] == "Sample Post"
    assert info["subtitle"] == "Test subtitle"
    assert info["date"] == date(2024, 5, 4)
    assert info["filename"] == "sample-post"
    assert info["tags"] == ["vision", "ai"]
    assert "blog-image" in info["content"]
    assert "/static/img/example.png" in info["content"]


def test_list_all_blog_info_sorts_by_descending_date(tmp_path, monkeypatch):
    write_post(tmp_path / "old.md", "Old", "2022-01-01")
    write_post(tmp_path / "new.md", "New", "2025-01-01")
    monkeypatch.setattr(app_module, "BLOG_DIR", str(tmp_path))

    posts = list_all_blog_info()

    assert [post["title"] for post in posts] == ["New", "Old"]


def test_filter_blog_posts_by_tag(tmp_path, monkeypatch):
    write_post(tmp_path / "vision.md", "Vision", "2024-01-01", "[vision, ai]")
    write_post(tmp_path / "writing.md", "Writing", "2024-01-02", "[writing]")
    monkeypatch.setattr(app_module, "BLOG_DIR", str(tmp_path))

    assert [post["title"] for post in filter_blog_posts_by_tag("vision")] == ["Vision"]
    assert filter_blog_posts_by_tag("unknown") == []


def test_format_date_handles_date_and_none():
    assert format_date(date(2024, 5, 4)) == "May 04, 2024"
    assert format_date(None) == "No Date"


def test_bulma_image_processor_renders_static_figure():
    class Match:
        def group(self, index):
            return {1: "Alt text", 2: "img/example.png"}[index]

        def start(self, index):
            return 0

        def end(self, index):
            return 31

    processor = BulmaImageProcessor(r"!\[([^\]]*)\]\(([^)]*)\)", None)

    with app.test_request_context():
        figure, start, end = processor.handleMatch(Match(), "")

    image = figure.find("img")
    assert figure.get("class") == "image"
    assert image.get("class") == "blog-image"
    assert image.get("alt") == "Alt text"
    assert image.get("src") == "/static/img/example.png"
    assert (start, end) == (0, 31)


def test_load_yaml_data_loads_home_data():
    data = load_yaml_data("home.yaml")

    assert data["profile"]["name"] == "Raluca-Maria Sandu"
    assert "timeline" not in data


def test_load_yaml_data_loads_shared_timeline_data():
    data = load_yaml_data("timeline.yaml")

    assert data["timeline"][0]["type"] == "work"
    assert data["timeline"][0]["title"] == "Machine Learning Engineer | Stealth AI Start-up"


def test_homepage_render_flow():
    response = app.test_client().get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Raluca-Maria Sandu" in html
    assert "rmsan@duck.com" in html
    assert "https://github.com/rmsandu" in html
    assert "Featured blog projects" in html
    assert "Experience &amp; Education" in html
    assert "Resume" not in html
    assert "Machine Learning Engineer" in html
    assert "multi-view character generation" in html
    assert "Machine Learning Team Lead" in html
    assert "7+ projects" in html
    assert "CSL" not in html
    assert "A brief timeline" not in html
    assert ">Path<" not in html


def test_blog_list_flow_orders_newest_first():
    response = app.test_client().get("/blogList.html")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert html.index("Segmentation of Fine Facial Wrinkles with U-Net") < html.index(
        "Generative AI Image Tools in Marketing and Design"
    )


def test_blog_tag_filter_flow():
    response = app.test_client().get("/search/blogList/segmentation.html")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Segmentation of Fine Facial Wrinkles with U-Net" in html
    assert "Fine-tuning with Stable Diffusion XL" not in html


def test_blog_post_flow():
    response = app.test_client().get("/blog/2025-04-18-wrinkle-segmentation.html")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Segmentation of Fine Facial Wrinkles with U-Net" in html
    assert "April 18, 2025" in html
    assert "Model architecture" in html


def test_publications_flow():
    client = app.test_client()
    response = client.get("/publications.html")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Volumetric quantitative ablation margins" in html
    assert 'href="/static/Biomedica_Sandu_2015.pdf"' in html
    assert "Conference paper" in html
    assert 'href="/static/Biomedica_Sandu_2015_Poster.pdf"' in html
    assert "Poster" in html
    assert "citation_for_view=5qskcz0AAAAJ:u5HHmVD_uO8C" not in html
    assert client.get("/static/Biomedica_Sandu_2015.pdf").status_code == 200
    assert client.get("/static/Biomedica_Sandu_2015_Poster.pdf").status_code == 200


def test_resume_page_removed():
    response = app.test_client().get("/resume.html")

    assert response.status_code == 404


def test_static_freeze_flow(tmp_path):
    app.config.update(
        FREEZER_DESTINATION=str(tmp_path),
        FREEZER_REMOVE_EXTRA_FILES=False,
        FREEZER_RELATIVE_URLS=True,
    )
    freezer = Freezer(app)

    @freezer.register_generator
    def render_blog_page():
        yield {"page_name": "2025-04-18-wrinkle-segmentation"}

    freezer.freeze()

    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "blogList.html").exists()
    assert (tmp_path / "publications.html").exists()
    assert not (tmp_path / "resume.html").exists()
    assert (tmp_path / "blog" / "2025-04-18-wrinkle-segmentation.html").exists()
