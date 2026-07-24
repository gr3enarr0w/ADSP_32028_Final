"""FAQ HTML parsing — avoids regex ReDoS by using stdlib HTMLParser."""

from html.parser import HTMLParser


class _FAQHTMLParser(HTMLParser):
    """Parse FAQ HTML back into structured data.

    Handles nested inline tags (strong, em, i, a, code) by tracking which
    *section* we're accumulating text for, rather than relying on the
    immediate tag. Also handles richer structures with multiple h3/h4
    sub-sections, ul/ol lists, etc.
    """

    def __init__(self):
        super().__init__()
        self.question = ""
        self.answer = ""
        self.steps: list[str] = []
        self.limitations = ""
        self._tag_stack: list[str] = []
        # Section tracking: "answer", "steps", "limitations", or None
        self._section: str | None = None
        self._got_answer = False
        self._current_li_parts: list[str] = []
        self._current_text_parts: list[str] = []

    def _outer_tag(self) -> str | None:
        """Return the nearest block-level ancestor tag."""
        for tag in reversed(self._tag_stack):
            if tag in ("h2", "h3", "h4", "p", "li", "ul", "ol"):
                return tag
        return None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._tag_stack.append(tag)
        if tag == "li":
            self._current_li_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._current_li_parts:
            step_text = "".join(self._current_li_parts).strip()
            if step_text:
                self.steps.append(step_text)
            self._current_li_parts = []

        if tag == "p" and self._current_text_parts:
            text = "".join(self._current_text_parts).strip()
            if text:
                if self._section == "limitations":
                    if self.limitations:
                        self.limitations += " " + text
                    else:
                        self.limitations = text
                elif self._section == "steps":
                    # Paragraph between list items — append as a step
                    self.steps.append(text)
                elif not self._got_answer:
                    self.answer = text
                    self._got_answer = True
                    self._section = "answer"
                else:
                    # Additional paragraph content — append as a step
                    self.steps.append(text)
            self._current_text_parts = []

        # Pop tag stack (handle mismatched tags gracefully)
        if self._tag_stack:
            try:
                idx = len(self._tag_stack) - 1 - self._tag_stack[::-1].index(tag)
                self._tag_stack.pop(idx)
            except ValueError:
                pass

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return

        outer = self._outer_tag()

        if outer == "h2":
            self.question += text
        elif outer in ("h3", "h4"):
            lower = text.lower()
            if "limitation" in lower:
                self._section = "limitations"
            elif any(kw in lower for kw in ("step", "how to", "pitfall", "troubleshoot",
                                             "follow-up", "common", "why")):
                self._section = "steps"
            else:
                self._section = "steps"
        elif outer == "li":
            self._current_li_parts.append(data)
        elif outer == "p":
            self._current_text_parts.append(data)
        else:
            # Text outside known block tags — accumulate based on section
            if self._section == "limitations":
                if self.limitations:
                    self.limitations += " " + text
                else:
                    self.limitations = text


def parse_faq_html(body_html: str) -> dict:
    """Parse FAQ HTML into structured dict using HTMLParser."""
    parser = _FAQHTMLParser()
    parser.feed(body_html)
    return {
        "question": parser.question,
        "answer": parser.answer,
        "steps": parser.steps,
        "known_limitations": parser.limitations,
    }
