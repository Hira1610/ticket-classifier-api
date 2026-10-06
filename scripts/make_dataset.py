"""Generates a SYNTHETIC support-ticket dataset (data/tickets.csv).

The data is template-based and artificial: it is meant for demonstrating the ML pipeline,
not for claiming real-world accuracy. Replace it with real data to get meaningful results.
"""
import csv
import random

random.seed(42)

CATEGORIES = {
    "billing": {
        "subjects": ["invoice", "payment", "refund", "charge", "subscription fee", "billing statement"],
        "templates": [
            "I was charged twice for my {s} this month",
            "Please send me a copy of my {s}",
            "My {s} shows the wrong amount",
            "I want a refund for the {s}",
            "Why did my {s} increase without notice",
            "Unable to update the card used for my {s}",
        ],
    },
    "technical": {
        "subjects": ["app", "website", "login page", "dashboard", "mobile app", "API"],
        "templates": [
            "The {s} keeps crashing when I open it",
            "I get an error message on the {s} after the update",
            "The {s} is very slow and sometimes does not load",
            "Cannot upload files, the {s} shows a blank screen",
            "Bug: the {s} freezes when I click save",
            "The {s} returns a 500 server error",
        ],
    },
    "account": {
        "subjects": ["password", "email address", "profile", "account", "two-factor code", "username"],
        "templates": [
            "I forgot my {s} and cannot sign in",
            "How do I change my {s}",
            "My {s} was locked after several attempts",
            "Please delete my {s} permanently",
            "I did not receive the verification for my {s}",
            "I need to recover my {s}",
        ],
    },
    "shipping": {
        "subjects": ["order", "package", "delivery", "parcel", "tracking number", "shipment"],
        "templates": [
            "My {s} has not arrived yet",
            "The {s} was delivered to the wrong address",
            "Where is my {s}, it is already late",
            "The {s} arrived damaged",
            "Can I change the address for my {s}",
            "The {s} status has not updated for a week",
        ],
    },
}
FILLERS = ["", "Hello, ", "Hi team, ", "Urgent: ", "Please help. ", "Good morning, "]
ENDINGS = ["", " Thanks.", " Please advise.", " This is the second time.", " I need this fixed soon."]

rows = []
for label, spec in CATEGORIES.items():
    for _ in range(60):
        text = random.choice(FILLERS) + random.choice(spec["templates"]).format(s=random.choice(spec["subjects"])) + random.choice(ENDINGS)
        rows.append((text.strip(), label))
random.shuffle(rows)
with open("data/tickets.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    w.writerows(rows)
print(f"Wrote {len(rows)} rows")
