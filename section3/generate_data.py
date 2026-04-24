from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


LABELS = [
    "billing",
    "technical_issue",
    "feature_request",
    "complaint",
    "other",
]

TRAIN_EXAMPLES_PER_CLASS = 200
EVAL_EXAMPLES_PER_CLASS = 25


TEMPLATES: Dict[str, List[str]] = {
    "billing": [
        "I was charged {count} times for my {plan} plan in {month}.",
        "Please refund the extra invoice for {service}; the total is wrong.",
        "My card was billed after I already canceled {service}.",
        "Why did I receive a {currency}{amount} charge for {service} this week?",
        "The invoice for {month} shows tax that should not be there.",
    ],
    "technical_issue": [
        "The {feature} button does nothing on {platform}.",
        "I keep getting the error '{error}' when I try to {action}.",
        "The app crashes every time I open the {feature} page.",
        "Login fails on {platform} after I enter the OTP.",
        "Uploads are stuck at {percent}% and never finish.",
    ],
    "feature_request": [
        "Please add a way to {action} from the {feature} screen.",
        "It would help if the product supported {feature} for teams.",
        "Can you build an option to export data in {format}?",
        "I want a dashboard that shows {metric} over time.",
        "Please support integrations with {integration}.",
    ],
    "complaint": [
        "Your service has been frustrating all week and nobody has helped me.",
        "I am unhappy with how support handled my case about {feature}.",
        "This is unacceptable; I keep paying for a product that feels unreliable.",
        "I have contacted support {count} times and still do not have a resolution.",
        "The experience is disappointing and your team keeps closing my tickets.",
    ],
    "other": [
        "I want to know whether your company offers internships in {month}.",
        "Can you share documentation about your security certifications?",
        "What is the best email address for your partnership team?",
        "Do you have office hours for sales demos in {region}?",
        "I am just checking whether this message reached the right inbox.",
    ],
}


SLOTS = {
    "count": ["two", "three", "four", "five"],
    "plan": ["starter", "growth", "business", "annual"],
    "month": ["January", "February", "March", "April", "May", "June"],
    "service": ["subscription", "workspace", "analytics add-on", "team license"],
    "currency": ["$", "₹", "€"],
    "amount": ["9.99", "29", "49", "99", "199"],
    "feature": ["export", "reports", "settings", "billing", "notifications"],
    "platform": ["Chrome", "Safari", "Firefox", "Android", "iPhone"],
    "error": ["Server unavailable", "Access denied", "Unexpected token", "Timeout reached"],
    "action": ["download a CSV", "reset my password", "submit a form", "invite a teammate"],
    "percent": ["45", "67", "82", "99"],
    "format": ["CSV", "XLSX", "PDF", "JSON"],
    "metric": ["usage by team", "failed jobs", "monthly spend", "response time"],
    "integration": ["Slack", "Salesforce", "Zendesk", "HubSpot"],
    "region": ["India", "EMEA", "North America", "APAC"],
}


MANUAL_EVAL: Dict[str, List[str]] = {
    "billing": [
        "I canceled last week, but my card was still charged today.",
        "The invoice amount is higher than the quoted annual price.",
        "Why is there a duplicate charge for my March subscription?",
        "I upgraded once, but it looks like I was billed for both plans.",
        "The receipt shows tax twice on the same order.",
        "Please explain why my refund has not appeared yet.",
        "I need help correcting an overcharge on our team account.",
        "My payment failed yesterday, but the amount was still deducted.",
        "We were invoiced for 12 seats even though we only use 8.",
        "The renewal charge is wrong compared with the contract.",
        "Our finance team needs a corrected invoice for last month.",
        "I was charged after downgrading to the cheaper plan.",
        "The billing portal shows an unpaid amount that we already settled.",
        "Can you reverse the duplicate debit on my corporate card?",
        "Why was I charged in dollars instead of rupees?",
        "The trial should have been free, but I see a payment on my statement.",
        "I need the GST number added to an existing invoice.",
        "The coupon was accepted during checkout, but the final charge ignored it.",
        "A refund was promised, but no money has come back yet.",
        "My account was suspended even though the invoice was paid.",
        "Please send a revised invoice with the correct company name.",
        "The billing cycle changed without warning and now the amount is off.",
        "I was charged for add-ons that are not enabled in our workspace.",
        "The pro-rated amount on the upgrade invoice looks incorrect.",
        "Can someone audit the charges on our April statement?",
    ],
    "technical_issue": [
        "The mobile app freezes on the login screen after the OTP step.",
        "Exporting a report to CSV fails with an internal server error.",
        "The dashboard loads forever and never shows the graphs.",
        "I cannot upload attachments from Firefox anymore.",
        "The password reset link opens a blank page.",
        "Notifications stopped arriving after the latest release.",
        "Our users are seeing a 500 error on the checkout page.",
        "Clicking save in settings does nothing at all.",
        "The API returns malformed JSON for the usage endpoint.",
        "Search is timing out whenever I query older records.",
        "The Android app crashes when I open billing history.",
        "I cannot invite teammates because the modal never appears.",
        "The website logs me out every few minutes.",
        "Uploads fail at 99 percent for large PDF files.",
        "Two-factor authentication codes are being rejected incorrectly.",
        "The report preview shows corrupted characters.",
        "Our SSO login loop never completes.",
        "The table filter breaks when I select multiple statuses.",
        "Pages are rendering without CSS in Safari.",
        "Webhook deliveries are delayed and sometimes duplicated.",
        "The app hangs after I click submit on the feedback form.",
        "The audit log page returns access denied for admins.",
        "My browser says the download link is invalid.",
        "The analytics chart is missing data from yesterday.",
        "The retry button spins forever and does not recover.",
    ],
    "feature_request": [
        "Please add dark mode for the admin dashboard.",
        "We need an option to export tickets directly to Excel.",
        "Can you support scheduled report emails every Monday?",
        "It would be useful to tag teammates in internal notes.",
        "Please add a bulk archive action for completed tasks.",
        "I want custom roles with granular permissions.",
        "Could you build a native Slack integration?",
        "Please support webhook retries with configurable backoff.",
        "A calendar view for project milestones would help us.",
        "Can you add filters for invoices by department?",
        "We need the ability to rename default ticket categories.",
        "It would be helpful to save report templates.",
        "Please allow CSV imports for user lists.",
        "Can the product support multiple approvers in one workflow?",
        "I would like a read-only guest access mode.",
        "Please add keyboard shortcuts for faster navigation.",
        "A side-by-side document compare view would be great.",
        "Can you provide an API endpoint for audit logs?",
        "I want usage alerts when we cross a spending threshold.",
        "Please support attachments larger than 25 MB.",
        "It would help to pin frequently used dashboards.",
        "Can you add localized date formats per workspace?",
        "Please let us duplicate an existing automation rule.",
        "A built-in satisfaction survey after ticket closure would be useful.",
        "Could you add conditional form fields for enterprise workflows?",
    ],
    "complaint": [
        "I am tired of chasing support for a simple answer.",
        "This has been a terrible experience from start to finish.",
        "Your team keeps closing my tickets without fixing anything.",
        "I am extremely disappointed with how this issue was handled.",
        "The product feels unreliable and your responses are too slow.",
        "Nobody seems to take ownership of my case.",
        "I should not have to contact support four times for this.",
        "The service quality has dropped a lot in the past month.",
        "This level of support is unacceptable for a paid plan.",
        "I am frustrated that no one follows through on promised updates.",
        "The tone from your support team was dismissive and unhelpful.",
        "I regret upgrading because the experience has become worse.",
        "Your process keeps wasting our team’s time.",
        "It should not be this hard to get a straight answer.",
        "The overall experience has been disappointing.",
        "We have lost confidence in your support organization.",
        "I keep getting generic replies instead of real help.",
        "This issue has dragged on far too long.",
        "I do not feel like anyone is listening to the problem.",
        "The repeated delays are incredibly frustrating.",
        "Your escalation process is broken and opaque.",
        "I expected better service at this price point.",
        "The communication on this ticket has been poor.",
        "My case was mishandled and then ignored.",
        "This product has caused more frustration than value lately.",
    ],
    "other": [
        "Can you send me your SOC 2 report request process?",
        "I would like to know whether you have a partner program.",
        "What is the contact email for media inquiries?",
        "Do you offer onboarding sessions for new customers?",
        "Where can I read about your privacy commitments?",
        "Can someone share your standard response time policy?",
        "I am looking for your public API documentation.",
        "Do you provide student discounts?",
        "What regions do you currently operate in?",
        "Can you point me to your legal terms page?",
        "I need your registered business address for procurement.",
        "Is there a webinar recording about your latest release?",
        "Who should we contact for reseller discussions?",
        "Where do I find your DPA template?",
        "Do you have a roadmap webinar next quarter?",
        "Can you confirm whether phone support is available?",
        "I want information about accessibility compliance.",
        "Where can I subscribe to product update emails?",
        "Does your company provide implementation consulting?",
        "I am checking whether this support inbox is monitored on weekends.",
        "Can you share your office hours for enterprise prospects?",
        "Where should I send a press request?",
        "Do you publish uptime reports publicly?",
        "How can I request a sales demo for our leadership team?",
        "Please share the best contact for procurement paperwork.",
    ],
}


def fill_template(template: str, rng: random.Random) -> str:
    values = {}
    for key, options in SLOTS.items():
        token = "{" + key + "}"
        if token in template:
            values[key] = rng.choice(options)
    return template.format(**values)


def build_train_set(seed: int = 7) -> List[dict]:
    rng = random.Random(seed)
    rows: List[dict] = []
    for label in LABELS:
        templates = TEMPLATES[label]
        for _ in range(TRAIN_EXAMPLES_PER_CLASS):
            rows.append({"text": fill_template(rng.choice(templates), rng), "label": label})
    rng.shuffle(rows)
    return rows


def build_eval_set() -> List[dict]:
    rows: List[dict] = []
    for label in LABELS:
        examples = MANUAL_EVAL[label]
        if len(examples) != EVAL_EXAMPLES_PER_CLASS:
            raise ValueError(f"{label} must contain exactly {EVAL_EXAMPLES_PER_CLASS} eval examples.")
        rows.extend({"text": text, "label": label} for text in examples)
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("section3/data"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = build_train_set()
    eval_rows = build_eval_set()

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)

    summary = {
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "labels": LABELS,
        "notes": {
            "train": "Synthetic template-generated examples for fine-tuning.",
            "eval": "Manually written evaluation examples for held-out testing.",
        },
    }
    with open(args.output_dir / "dataset_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
