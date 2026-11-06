"""
generate_dataset.py
--------------------
Generates a synthetic but linguistically realistic labeled news dataset
(data/news_dataset.csv) — an OPTIONAL fallback if you want a small,
fast-to-train demo dataset instead of the bundled real dataset.

The project ships with a REAL dataset by default (fake_or_real_news.csv,
6,334 genuine English news articles). Only run this script if you
specifically want to swap in the synthetic version instead.
"""

import csv
import random

random.seed(42)

TOPICS = [
    "the economy", "a new vaccine", "a celebrity marriage", "the national election",
    "climate policy", "a tech company merger", "a local sports team", "a viral health trend",
    "a government scandal", "a scientific discovery", "cryptocurrency markets", "a school policy change",
    "a natural disaster", "an airline safety issue", "a food safety recall", "a space mission",
]

REAL_TEMPLATES = [
    "According to officials, {topic} was discussed during a press briefing on Tuesday, with details expected to follow after further review.",
    "A report published by researchers this week examined {topic}, citing data collected over the past two years.",
    "City officials confirmed that {topic} will be addressed in the upcoming quarterly meeting, according to a public statement.",
    "The department released a statement clarifying its position on {topic}, noting that further updates would be shared as they become available.",
    "Data from the latest survey suggests {topic} has changed moderately compared to last year, analysts said.",
    "In an interview, the spokesperson addressed questions regarding {topic}, providing context based on verified figures.",
    "Officials have not yet confirmed all details surrounding {topic}, pending an ongoing internal review.",
    "The committee is expected to release a full report on {topic} later this month after consulting with independent experts.",
    "Local authorities issued a routine advisory concerning {topic}, urging residents to stay informed through official channels.",
    "Experts caution that early findings on {topic} should be interpreted carefully until peer review is complete.",
]

FAKE_TEMPLATES = [
    "BREAKING: You won't believe what they're hiding about {topic}!!! Share before this gets DELETED!",
    "SHOCKING truth about {topic} that the mainstream media REFUSES to tell you — click now!",
    "Doctors HATE this one secret about {topic} that insiders don't want you to know!",
    "URGENT: {topic} is a total HOAX, and here's the proof they don't want you to see!",
    "This will change EVERYTHING you thought you knew about {topic} — number 3 will shock you!",
    "Anonymous insider LEAKS terrifying details about {topic}; government scrambles to cover it up!",
    "You need to see this NOW: {topic} exposed in viral video that's breaking the internet!",
    "WAKE UP! {topic} is proof of a massive conspiracy that's been hidden for YEARS!",
    "Scientists are STUNNED after this discovery about {topic} that defies all logic!!!",
    "They don't want you to read this: the REAL story behind {topic}, finally revealed!",
]

REAL_SOURCES = ["Reuters", "Associated Press", "BBC News", "local news desk", "official press release", "staff reporter"]
FAKE_SOURCES = ["a friend of a friend", "an anonymous forum post", "a viral WhatsApp forward", "an unnamed insider", "a random blog"]


def make_row(is_fake: bool) -> dict:
    topic = random.choice(TOPICS)
    if is_fake:
        text = random.choice(FAKE_TEMPLATES).format(topic=topic)
        text += f" (Source: {random.choice(FAKE_SOURCES)})"
        label = 1
    else:
        text = random.choice(REAL_TEMPLATES).format(topic=topic)
        text += f" (Source: {random.choice(REAL_SOURCES)})"
        label = 0
    return {"text": text, "label": label}


def main(n_per_class: int = 300, out_path: str = "data/news_dataset.csv"):
    rows = [make_row(True) for _ in range(n_per_class)] + [make_row(False) for _ in range(n_per_class)]
    random.shuffle(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
