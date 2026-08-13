from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml

from image_gen.webui.theme.security import validate_svg_visual_content

HELP_CENTER_SCHEMA = "image-gen-help-center-v1"
HELP_CENTER_CONTRACT_VERSION = 1
HELP_SEARCH_MIN_CHARS = 3
HELP_SEARCH_LIMIT = 12

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
_VIDEO_SUFFIXES = {".mp4", ".webm", ".ogg"}
_MEDIA_SUFFIXES = _IMAGE_SUFFIXES | _VIDEO_SUFFIXES
_MARKDOWN_SUFFIX = ".md"
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_MARKDOWN_NOISE = re.compile(r"[`*_>#\[\](){}|]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class HelpTopic:
    topic_id: str
    source_path: Path
    title: str
    summary: str
    category_path: tuple[str, ...]
    keywords: tuple[str, ...]
    related: tuple[str, ...]
    media: tuple[dict[str, Any], ...]
    external_links: tuple[dict[str, str], ...]
    featured: bool
    markdown: str

    def catalog_dict(self) -> dict[str, Any]:
        return {
            "id": self.topic_id,
            "title": self.title,
            "summary": self.summary,
            "categoryPath": list(self.category_path),
            "keywords": list(self.keywords),
            "related": list(self.related),
            "featured": self.featured,
        }


class HelpCenterService:
    """Read public user help from the repository's ``help_documentation`` root only."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.help_root = (self.project_root / "help_documentation").resolve()

    @staticmethod
    def _normalize_topic_id(value: str) -> str:
        raw = str(value or "").strip().replace("\\", "/")
        if raw.endswith(_MARKDOWN_SUFFIX):
            raw = raw[: -len(_MARKDOWN_SUFFIX)]
        candidate = PurePosixPath(raw)
        if not raw or candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("Help topic ID is invalid.")
        return candidate.as_posix()

    def _resolved_inside_help(self, relative: str | PurePosixPath) -> Path:
        rel = PurePosixPath(str(relative).replace("\\", "/"))
        if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
            raise ValueError("Help content path is invalid.")
        resolved = (self.help_root / Path(*rel.parts)).resolve()
        try:
            resolved.relative_to(self.help_root)
        except ValueError as exc:
            raise ValueError("Help content path escapes the public help root.") from exc
        return resolved

    @staticmethod
    def _front_matter(text: str) -> tuple[dict[str, Any], str]:
        match = _FRONT_MATTER.match(text)
        if not match:
            return {}, text
        try:
            metadata = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid help document front matter: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Help document front matter must be a mapping.")
        return dict(metadata), text[match.end() :]

    @staticmethod
    def _heading_title(markdown: str, fallback: str) -> str:
        for line in markdown.splitlines():
            match = re.match(r"^\s*#\s+(.+?)\s*#*\s*$", line)
            if match:
                return match.group(1).strip()
        return fallback

    @staticmethod
    def _summary(markdown: str) -> str:
        paragraphs: list[str] = []
        for line in markdown.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("```"):
                if paragraphs:
                    break
                continue
            if stripped.startswith(("- ", "* ", "> ")):
                continue
            paragraphs.append(stripped)
        value = " ".join(paragraphs).strip()
        value = _MARKDOWN_NOISE.sub("", value)
        value = _WHITESPACE.sub(" ", value).strip()
        return value[:240]

    @staticmethod
    def _string_list(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        values = value if isinstance(value, list) else [value]
        return tuple(str(item).strip() for item in values if str(item).strip())

    def _normalize_media(self, topic_id: str, topic_path: Path, value: Any) -> tuple[dict[str, Any], ...]:
        items = value if isinstance(value, list) else ([] if value is None or value == "" else [value])
        output: list[dict[str, Any]] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            media_type = str(raw.get("type") or "image").strip().lower()
            src = str(raw.get("src") or "").strip().replace("\\", "/")
            if media_type not in {"image", "video"} or not src:
                continue
            parsed = urlparse(src)
            if parsed.scheme or parsed.netloc:
                # External content is intentionally link-only. It must be declared
                # through external_links so opening help never silently contacts a third party.
                continue
            try:
                media_path = (topic_path.parent / Path(*PurePosixPath(src).parts)).resolve()
                canonical_relative = media_path.relative_to(self.help_root).as_posix()
            except (OSError, ValueError):
                continue
            suffix = media_path.suffix.lower()
            if suffix not in _MEDIA_SUFFIXES:
                continue
            if media_type == "image" and suffix not in _IMAGE_SUFFIXES:
                continue
            if media_type == "video" and suffix not in _VIDEO_SUFFIXES:
                continue
            if not media_path.is_file():
                continue
            if suffix == ".svg":
                validation = validate_svg_visual_content(media_path.read_text(encoding="utf-8"))
                if not validation.valid:
                    continue
            output.append({
                "type": media_type,
                "src": f"/api/help/media/{canonical_relative}",
                "alt": str(raw.get("alt") or "").strip(),
                "caption": str(raw.get("caption") or "").strip(),
                "poster": str(raw.get("poster") or "").strip(),
            })
        return tuple(output)

    @staticmethod
    def _external_links(value: Any) -> tuple[dict[str, str], ...]:
        items = value if isinstance(value, list) else ([] if value is None or value == "" else [value])
        links: list[dict[str, str]] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            href = str(raw.get("href") or "").strip()
            parsed = urlparse(href)
            if parsed.scheme.lower() != "https" or not parsed.netloc:
                continue
            label = str(raw.get("label") or raw.get("title") or parsed.netloc).strip()
            links.append({"label": label, "href": href})
        return tuple(links)

    def _load_topic(self, path: Path) -> HelpTopic:
        relative = path.resolve().relative_to(self.help_root).as_posix()
        topic_id = relative[: -len(_MARKDOWN_SUFFIX)]
        raw = path.read_text(encoding="utf-8")
        metadata, markdown = self._front_matter(raw)
        fallback_title = path.stem.replace("_", " ").replace("-", " ").title()
        title = str(metadata.get("title") or self._heading_title(markdown, fallback_title)).strip()
        summary = str(metadata.get("summary") or self._summary(markdown)).strip()
        category_path = tuple(part.replace("_", " ").replace("-", " ").title() for part in PurePosixPath(topic_id).parent.parts if part != ".")
        keywords = self._string_list(metadata.get("keywords"))
        related = tuple(
            self._normalize_topic_id(item)
            for item in self._string_list(metadata.get("related"))
            if item
        )
        return HelpTopic(
            topic_id=topic_id,
            source_path=path,
            title=title,
            summary=summary,
            category_path=category_path,
            keywords=keywords,
            related=related,
            media=self._normalize_media(topic_id, path, metadata.get("media")),
            external_links=self._external_links(metadata.get("external_links")),
            featured=bool(metadata.get("featured", False)),
            markdown=markdown.strip() + "\n",
        )

    def topics(self) -> dict[str, HelpTopic]:
        if not self.help_root.is_dir():
            return {}
        output: dict[str, HelpTopic] = {}
        for path in sorted(self.help_root.rglob("*.md")):
            try:
                path.resolve().relative_to(self.help_root)
                topic = self._load_topic(path)
            except (OSError, UnicodeError, ValueError):
                continue
            output[topic.topic_id] = topic
        return output

    @staticmethod
    def _tree(topics: dict[str, HelpTopic]) -> list[dict[str, Any]]:
        root: dict[str, Any] = {"children": {}, "topics": []}
        for topic in topics.values():
            if topic.topic_id == "index":
                continue
            node = root
            for depth, part in enumerate(PurePosixPath(topic.topic_id).parent.parts):
                if part == ".":
                    continue
                children = node["children"]
                if part not in children:
                    children[part] = {
                        "id": "/".join(PurePosixPath(topic.topic_id).parent.parts[: depth + 1]),
                        "title": part.replace("_", " ").replace("-", " ").title(),
                        "children": {},
                        "topics": [],
                    }
                node = children[part]
            if PurePosixPath(topic.topic_id).name == "index":
                node["landingTopicId"] = topic.topic_id
                node["title"] = topic.title
            else:
                node["topics"].append(topic.catalog_dict())

        def finalize(node: dict[str, Any]) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for child in sorted(node["children"].values(), key=lambda item: str(item["title"]).casefold()):
                result.append({
                    "id": child["id"],
                    "title": child["title"],
                    "landingTopicId": child.get("landingTopicId", ""),
                    "topics": sorted(child["topics"], key=lambda item: str(item["title"]).casefold()),
                    "children": finalize(child),
                })
            return result

        return finalize(root)

    def catalog(self) -> dict[str, Any]:
        topics = self.topics()
        featured = [topic.catalog_dict() for topic in topics.values() if topic.featured]
        if not featured:
            featured = [topic.catalog_dict() for topic in list(topics.values())[:6]]
        return {
            "schema": HELP_CENTER_SCHEMA,
            "contractVersion": HELP_CENTER_CONTRACT_VERSION,
            "minimumSearchCharacters": HELP_SEARCH_MIN_CHARS,
            "rootTopicId": "index" if "index" in topics else "",
            "topicCount": len(topics),
            "tree": self._tree(topics),
            "featured": featured[:8],
        }

    def topic(self, topic_id: str) -> dict[str, Any]:
        normalized = self._normalize_topic_id(topic_id)
        topics = self.topics()
        topic = topics.get(normalized)
        if topic is None:
            raise FileNotFoundError(f"Help topic '{normalized}' was not found.")
        related = [topics[item].catalog_dict() for item in topic.related if item in topics and item != normalized]
        if len(related) < 3:
            candidates = [
                item for item in topics.values()
                if item.topic_id != normalized
                and item.catalog_dict() not in related
                and item.category_path == topic.category_path
            ]
            for item in candidates:
                if len(related) >= 4:
                    break
                related.append(item.catalog_dict())
        return {
            "schema": HELP_CENTER_SCHEMA,
            "contractVersion": HELP_CENTER_CONTRACT_VERSION,
            "topic": {
                **topic.catalog_dict(),
                "markdown": topic.markdown,
                "media": list(topic.media),
                "externalLinks": list(topic.external_links),
                "relatedTopics": related[:6],
            },
        }

    def search(self, query: str) -> dict[str, Any]:
        normalized = _WHITESPACE.sub(" ", str(query or "").strip().casefold())
        if len(normalized) < HELP_SEARCH_MIN_CHARS:
            return {
                "schema": HELP_CENTER_SCHEMA,
                "minimumSearchCharacters": HELP_SEARCH_MIN_CHARS,
                "query": str(query or ""),
                "results": [],
            }
        tokens = tuple(token for token in normalized.split(" ") if token)
        scored: list[tuple[int, HelpTopic, str]] = []
        for topic in self.topics().values():
            body = _WHITESPACE.sub(" ", _MARKDOWN_NOISE.sub(" ", topic.markdown).casefold())
            title = topic.title.casefold()
            summary = topic.summary.casefold()
            keywords = " ".join(topic.keywords).casefold()
            category = " ".join(topic.category_path).casefold()
            haystack = " ".join((title, summary, keywords, category, body))
            if not all(token in haystack for token in tokens):
                continue
            score = 0
            if normalized in title:
                score += 120
            if any(token in title for token in tokens):
                score += 60
            if normalized in keywords:
                score += 50
            if normalized in summary:
                score += 35
            if normalized in body:
                score += 15
            if topic.featured:
                score += 3
            snippet_source = topic.summary or self._summary(topic.markdown)
            scored.append((score, topic, snippet_source))
        scored.sort(key=lambda item: (-item[0], item[1].title.casefold()))
        return {
            "schema": HELP_CENTER_SCHEMA,
            "minimumSearchCharacters": HELP_SEARCH_MIN_CHARS,
            "query": str(query or ""),
            "results": [
                {**topic.catalog_dict(), "snippet": snippet}
                for _score, topic, snippet in scored[:HELP_SEARCH_LIMIT]
            ],
        }

    def media_path(self, relative_path: str) -> Path:
        path = self._resolved_inside_help(relative_path)
        if path.suffix.lower() not in _MEDIA_SUFFIXES or not path.is_file():
            raise FileNotFoundError("Help media was not found.")
        if path.suffix.lower() == ".svg":
            validation = validate_svg_visual_content(path.read_text(encoding="utf-8"))
            if not validation.valid:
                raise ValueError("Help SVG failed safety validation.")
        return path
