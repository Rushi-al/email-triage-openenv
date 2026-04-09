"""
email_dataset.py - Realistic support email corpus with ground-truth labels.

Each email has:
  - Metadata (id, subject, sender, timestamp)
  - Body text
  - ground_truth urgency
  - ground_truth department
  - A sample ideal response (used to score Task 3)
  - Key phrases the ideal response should contain
"""

from typing import List, Dict, Any

EMAILS: List[Dict[str, Any]] = [
    # URGENT / BILLING
    {
        "email_id": "E001",
        "subject": "CHARGED TWICE - PLEASE FIX IMMEDIATELY",
        "sender": "john.doe@example.com",
        "timestamp": "2024-03-15T09:14:00Z",
        "body": (
            "Hi, I just noticed that my credit card was charged $89.99 TWICE this month "
            "for my subscription. I only signed up once and I'm really angry about this. "
            "My account number is #4521. Please refund the duplicate charge TODAY - "
            "I cannot wait for a week-long investigation. This is completely unacceptable."
        ),
        "ground_truth": {
            "urgency": "urgent",
            "department": "billing",
            "response_keywords": ["apologize", "refund", "duplicate", "24", "account"],
            "ideal_response": (
                "Dear John,\n\nThank you for bringing this to our attention. I sincerely apologize "
                "for the duplicate charge on your account. I have escalated this to our billing "
                "team and a full refund for the duplicate $89.99 charge will be processed within "
                "24 hours. You will receive a confirmation email once completed.\n\n"
                "We are sorry for this inconvenience.\n\nBest regards,\nSupport Team"
            ),
        },
    },
    # URGENT / TECHNICAL
    {
        "email_id": "E002",
        "subject": "Production system DOWN - urgent",
        "sender": "ops-team@acmecorp.com",
        "timestamp": "2024-03-15T11:30:00Z",
        "body": (
            "Our entire production environment is completely down since 11:15 AM. "
            "We are losing $10,000 per minute. Your API is returning 503 errors on "
            "ALL endpoints. This is a P0 incident. We need an engineer on the phone "
            "within the next 15 minutes. Account: ACME-CORP-ENTERPRISE. "
            "This is mission critical - please escalate NOW."
        ),
        "ground_truth": {
            "urgency": "urgent",
            "department": "technical",
            "response_keywords": ["escalate", "engineer", "P0", "immediate", "outage"],
            "ideal_response": (
                "Dear ACME Corp Operations Team,\n\nI have immediately escalated this as a P0 "
                "critical incident. An on-call engineer will contact you within 15 minutes. "
                "Our incident response team is investigating the 503 errors on your account "
                "ACME-CORP-ENTERPRISE right now.\n\nWe understand the severity and business "
                "impact. You will receive updates every 10 minutes.\n\nBest regards,\nSupport Team"
            ),
        },
    },
    # NORMAL / TECHNICAL
    {
        "email_id": "E003",
        "subject": "How do I integrate the API with Python?",
        "sender": "dev.sarah@startup.io",
        "timestamp": "2024-03-14T14:22:00Z",
        "body": (
            "Hello! I'm trying to integrate your API into my Python application. "
            "I've read the docs but I'm getting a 401 Unauthorized error even though "
            "I'm passing the API key in the header. Here's my code:\n\n"
            "headers = {'Authorization': 'Bearer my_api_key'}\n"
            "requests.get('https://api.example.com/data', headers=headers)\n\n"
            "Could you let me know what I'm doing wrong? Not urgent but would love "
            "a response this week."
        ),
        "ground_truth": {
            "urgency": "normal",
            "department": "technical",
            "response_keywords": ["API key", "Authorization", "header", "documentation"],
            "ideal_response": (
                "Hi Sarah,\n\nThank you for reaching out! The 401 error is likely because the "
                "Authorization header format should use 'Api-Key' instead of 'Bearer'. "
                "Try: headers = {'Api-Key': 'your_api_key'}. If the issue persists, "
                "please check our Python SDK documentation at docs.example.com/python.\n\n"
                "Let me know if this resolves your issue!\n\nBest regards,\nTechnical Support"
            ),
        },
    },
    # NORMAL / BILLING
    {
        "email_id": "E004",
        "subject": "Question about my invoice for February",
        "sender": "mike.chen@business.com",
        "timestamp": "2024-03-13T10:05:00Z",
        "body": (
            "Hi team, I received my February invoice and noticed a line item for "
            "'Premium Support Add-on' at $25/month that I don't recall signing up for. "
            "Could you please clarify when this was added to my account and whether "
            "I can remove it? My account ID is BC-7823. Thanks in advance."
        ),
        "ground_truth": {
            "urgency": "normal",
            "department": "billing",
            "response_keywords": ["invoice", "add-on", "account", "remove", "clarify"],
            "ideal_response": (
                "Hi Mike,\n\nThank you for your inquiry about your February invoice. "
                "I've looked into account BC-7823 and can see the Premium Support Add-on "
                "was added on January 15th. I'd be happy to remove this and issue a "
                "prorated refund if you did not intentionally add this service. "
                "Please confirm and I'll process this right away.\n\nBest regards,\nBilling Team"
            ),
        },
    },
    # NORMAL / RETURNS
    {
        "email_id": "E005",
        "subject": "Return request - order #ORD-9821",
        "sender": "alice.wang@personal.net",
        "timestamp": "2024-03-14T16:48:00Z",
        "body": (
            "Hello, I'd like to return the laptop stand I ordered last week (order #ORD-9821). "
            "It arrived with a small crack on the right leg - I have photos if needed. "
            "I'd prefer a replacement rather than a refund. The product is otherwise "
            "great but the defect makes it unstable. Please let me know the return process."
        ),
        "ground_truth": {
            "urgency": "normal",
            "department": "returns",
            "response_keywords": ["return", "replacement", "defect", "photos", "shipping"],
            "ideal_response": (
                "Dear Alice,\n\nWe're sorry to hear your order #ORD-9821 arrived damaged. "
                "We'd be happy to send a replacement at no extra cost. Please email the "
                "photos to returns@example.com with your order number. We'll ship a "
                "replacement within 2-3 business days once confirmed and provide a "
                "prepaid return label for the damaged item.\n\nBest regards,\nReturns Team"
            ),
        },
    },
    # LOW / GENERAL
    {
        "email_id": "E006",
        "subject": "Do you have a mobile app?",
        "sender": "curious.user@gmail.com",
        "timestamp": "2024-03-12T08:30:00Z",
        "body": (
            "Hey, I've been using your service on desktop for a while now and love it. "
            "Was just wondering if you have a mobile app or if you plan to release one "
            "in the future? Would love to use this on the go. Thanks!"
        ),
        "ground_truth": {
            "urgency": "low",
            "department": "general",
            "response_keywords": ["mobile", "app", "roadmap", "thank"],
            "ideal_response": (
                "Hi there!\n\nThank you for being a valued user and for the kind words! "
                "We do have a mobile app available for iOS and Android - you can find it "
                "by searching 'ExampleApp' in the App Store or Google Play. "
                "We're always adding new features, so stay tuned for updates!\n\n"
                "Best regards,\nSupport Team"
            ),
        },
    },
    # LOW / GENERAL
    {
        "email_id": "E007",
        "subject": "Feature request: dark mode",
        "sender": "night.owl@devco.com",
        "timestamp": "2024-03-11T22:15:00Z",
        "body": (
            "Hi, long-time user here. I'd love to see a dark mode option in the dashboard. "
            "Working late at night and the bright white background is hard on the eyes. "
            "This is just a suggestion - no rush at all! Keep up the great work."
        ),
        "ground_truth": {
            "urgency": "low",
            "department": "general",
            "response_keywords": ["feature", "request", "roadmap", "feedback", "thank"],
            "ideal_response": (
                "Hi!\n\nThank you so much for the feedback and for being a loyal user! "
                "Dark mode is actually on our product roadmap. I've logged your request "
                "to help prioritize it. We'll announce it in our release notes when it "
                "ships. Thanks again for taking the time to write in!\n\nBest regards,\nProduct Team"
            ),
        },
    },
    # URGENT / RETURNS
    {
        "email_id": "E008",
        "subject": "Wrong item delivered - wedding gift URGENT",
        "sender": "megan.taylor@email.com",
        "timestamp": "2024-03-15T07:55:00Z",
        "body": (
            "I am absolutely furious. I ordered a crystal vase as a wedding gift "
            "(order #ORD-1144) and you sent me a completely different item - some kind "
            "of kitchen appliance. The wedding is THIS SATURDAY. I need the correct "
            "item delivered by Friday or this entire order refunded immediately. "
            "Please call me at 555-0192. This is a time-critical emergency."
        ),
        "ground_truth": {
            "urgency": "urgent",
            "department": "returns",
            "response_keywords": ["wrong item", "urgent", "expedite", "Saturday", "refund"],
            "ideal_response": (
                "Dear Megan,\n\nI am so sorry for this serious error with order #ORD-1144. "
                "I completely understand the urgency given the upcoming wedding. I am "
                "escalating this immediately for expedited shipping of the correct item. "
                "If we cannot guarantee delivery by Friday, we will issue a full refund "
                "today. A manager will call you at 555-0192 within the next 30 minutes.\n\n"
                "Sincerely,\nSenior Support Team"
            ),
        },
    },
    # NORMAL / GENERAL
    {
        "email_id": "E009",
        "subject": "Trouble logging into my account",
        "sender": "robert.smith@workemail.com",
        "timestamp": "2024-03-13T13:40:00Z",
        "body": (
            "Hello, I've been trying to log into my account for the past hour but keep "
            "getting an 'invalid credentials' error. I'm sure I'm using the right email "
            "and password - I even tried resetting the password twice but the reset "
            "emails aren't arriving. My account email is r.smith@workemail.com. "
            "Please help when you get a chance."
        ),
        "ground_truth": {
            "urgency": "normal",
            "department": "technical",
            "response_keywords": ["password", "reset", "email", "account", "spam"],
            "ideal_response": (
                "Hi Robert,\n\nThank you for reaching out. I've checked account "
                "r.smith@workemail.com and it appears active. The password reset emails "
                "may be going to your spam folder - please check there. If not found, "
                "I can manually trigger a reset from our end. Please also try clearing "
                "your browser cache. Let me know and I'll get this resolved for you!\n\n"
                "Best regards,\nTechnical Support"
            ),
        },
    },
    # LOW / BILLING
    {
        "email_id": "E010",
        "subject": "When does my subscription renew?",
        "sender": "janet.p@personalmail.org",
        "timestamp": "2024-03-10T15:20:00Z",
        "body": (
            "Hi there, just a quick question - I can't find where to see my subscription "
            "renewal date in the settings. Could you tell me when my current plan renews? "
            "My account email is janet.p@personalmail.org. No rush, thanks!"
        ),
        "ground_truth": {
            "urgency": "low",
            "department": "billing",
            "response_keywords": ["renewal", "subscription", "settings", "date", "account"],
            "ideal_response": (
                "Hi Janet!\n\nYour subscription renewal date can be found in Account Settings "
                "> Billing > Subscription. I've also checked and your plan renews on "
                "April 10th, 2024. You'll receive a reminder email 7 days before. "
                "Let me know if you have any other questions!\n\nBest regards,\nBilling Support"

        
            ),
        },
    },
        # ── URGENT / TECHNICAL ─────────────────────────────────────────────
    {
        "email_id": "E011",
        "subject": "Data breach detected - customer PII exposed",
        "sender": "security@fintech-corp.com",
        "timestamp": "2024-03-15T02:30:00Z",
        "body": (
            "CRITICAL SECURITY ALERT: Our monitoring system has detected unauthorized "
            "access to our customer database. Approximately 50,000 customer records "
            "including names, emails, and partial credit card numbers may be compromised. "
            "This happened 30 minutes ago. We need your security team IMMEDIATELY. "
            "Regulatory reporting deadline is in 4 hours. This is a legal emergency."
        ),
        "ground_truth": {
            "urgency": "urgent",
            "department": "technical",
            "response_keywords": ["security", "breach", "escalate", "immediate", "team"],
        },
    },
    # ── URGENT / BILLING ───────────────────────────────────────────────
    {
        "email_id": "E012",
        "subject": "Unauthorized charges - fraud on my account",
        "sender": "david.nguyen@gmail.com",
        "timestamp": "2024-03-15T08:20:00Z",
        "body": (
            "Someone has made 3 unauthorized purchases on my account totaling $847.50. "
            "I did NOT make these transactions. They happened last night while I was asleep. "
            "I need my account FROZEN immediately and all fraudulent charges reversed. "
            "My account ID is DN-2291. This is fraud and I will involve my bank and "
            "law enforcement if this is not resolved TODAY."
        ),
        "ground_truth": {
            "urgency": "urgent",
            "department": "billing",
            "response_keywords": ["fraud", "freeze", "refund", "investigate", "secure"],
        },
    },
    # ── NORMAL / RETURNS ───────────────────────────────────────────────
    {
        "email_id": "E013",
        "subject": "Package arrived but missing items",
        "sender": "priya.sharma@outlook.com",
        "timestamp": "2024-03-13T11:15:00Z",
        "body": (
            "Hi, I received my order #ORD-5521 today but it was missing 2 of the 4 items "
            "I ordered. The box was sealed so I don't think it was tampered with — it seems "
            "like a packing error. I ordered: 2x USB-C cables, 1x laptop stand, 1x mouse pad. "
            "Only the stand and mouse pad arrived. Can you send the missing cables or refund "
            "me for those 2 items? Order total was $67.99."
        ),
        "ground_truth": {
            "urgency": "normal",
            "department": "returns",
            "response_keywords": ["missing", "reship", "refund", "order", "apologize"],
        },
    },
    # ── NORMAL / BILLING ───────────────────────────────────────────────
    {
        "email_id": "E014",
        "subject": "Request to upgrade my subscription plan",
        "sender": "carlos.mendez@startup.co",
        "timestamp": "2024-03-14T09:45:00Z",
        "body": (
            "Hello, I'm currently on the Basic plan ($29/month) and would like to upgrade "
            "to the Professional plan ($79/month). I have 8 team members who need access "
            "and we're hitting the 5-user limit. Can you process the upgrade and let me know "
            "if there will be a prorated charge for the remainder of this month? "
            "Account: CM-4411. Thanks!"
        ),
        "ground_truth": {
            "urgency": "normal",
            "department": "billing",
            "response_keywords": ["upgrade", "prorated", "plan", "account", "team"],
        },
    },
    # ── LOW / TECHNICAL ────────────────────────────────────────────────
    {
        "email_id": "E015",
        "subject": "Question about API rate limits",
        "sender": "engineer@devteam.io",
        "timestamp": "2024-03-11T14:00:00Z",
        "body": (
            "Hi team, just a quick question — what are the rate limits for the REST API "
            "on the Standard tier? Our docs show 1000 requests/minute but I want to confirm "
            "before we design our system architecture. No urgency at all, just planning ahead "
            "for next quarter's project. Thanks!"
        ),
        "ground_truth": {
            "urgency": "low",
            "department": "technical",
            "response_keywords": ["rate limit", "API", "standard", "requests", "documentation"],
        },
    },
    # ── URGENT / RETURNS ───────────────────────────────────────────────
    {
        "email_id": "E016",
        "subject": "Defective medical device - urgent safety concern",
        "sender": "nurse.patricia@healthclinic.org",
        "timestamp": "2024-03-15T10:05:00Z",
        "body": (
            "We purchased 10 blood pressure monitors (order #ORD-8891) for our clinic "
            "and ALL of them are giving wildly inaccurate readings — up to 40mmHg off. "
            "We are using these on actual patients and this is a PATIENT SAFETY ISSUE. "
            "We have stopped using them immediately. We need a full refund AND replacements "
            "shipped overnight. This may require a product safety report. Please escalate NOW."
        ),
        "ground_truth": {
            "urgency": "urgent",
            "department": "returns",
            "response_keywords": ["safety", "escalate", "replacement", "refund", "urgent"],
        },
    },
    # ── NORMAL / GENERAL ───────────────────────────────────────────────
    {
        "email_id": "E017",
        "subject": "How do I export my data?",
        "sender": "maria.santos@company.com",
        "timestamp": "2024-03-13T15:30:00Z",
        "body": (
            "Hello! I'd like to export all my data from the platform — specifically my "
            "transaction history and account reports for the past 2 years. I need this "
            "for our annual audit. Is there a bulk export feature? If so, what formats "
            "are available (CSV, PDF, Excel)? Not urgent but would appreciate a response "
            "this week. Thank you!"
        ),
        "ground_truth": {
            "urgency": "normal",
            "department": "general",
            "response_keywords": ["export", "data", "format", "audit", "settings"],
        },
    },
    # ── LOW / GENERAL ──────────────────────────────────────────────────
    {
        "email_id": "E018",
        "subject": "Compliment — best customer service I've had",
        "sender": "happy.customer@email.com",
        "timestamp": "2024-03-10T11:00:00Z",
        "body": (
            "Hi! I just wanted to take a moment to say that your support team is absolutely "
            "fantastic. Last week your agent Sarah helped me resolve a billing issue in under "
            "10 minutes. I've dealt with many SaaS companies and this was the best experience "
            "I've had. Please pass on my thanks to her and the whole team. Keep up the great work!"
        ),
        "ground_truth": {
            "urgency": "low",
            "department": "general",
            "response_keywords": ["thank", "feedback", "team", "appreciate", "share"],
        },
    },
    # ── URGENT / BILLING ───────────────────────────────────────────────
    {
        "email_id": "E019",
        "subject": "Account suspended but I paid — business critical",
        "sender": "cfo@manufacturing-ltd.com",
        "timestamp": "2024-03-15T09:00:00Z",
        "body": (
            "Our account has been suspended due to a 'failed payment' but our bank confirms "
            "the payment of $4,200 cleared 3 days ago. Our entire team of 50 people cannot "
            "access the system and we are losing $5,000/hour in productivity. "
            "This is completely unacceptable. Please restore access IMMEDIATELY and explain "
            "why this happened. Account: MFG-CORP-001. I will be escalating to our legal team "
            "if not resolved within 1 hour."
        ),
        "ground_truth": {
            "urgency": "urgent",
            "department": "billing",
            "response_keywords": ["suspend", "restore", "payment", "access", "immediate"],
        },
    },
    # ── NORMAL / TECHNICAL ─────────────────────────────────────────────
    {
        "email_id": "E020",
        "subject": "Webhook not firing for specific events",
        "sender": "backend-dev@techstartup.com",
        "timestamp": "2024-03-14T16:20:00Z",
        "body": (
            "Hi, our webhook endpoint is receiving most events correctly but we've noticed "
            "that 'payment.refunded' events are never arriving. We've tested with your "
            "webhook tester and it shows 'delivered' but our server logs show nothing. "
            "Our endpoint is https://api.oursite.com/webhooks and it works for all other "
            "event types. This is affecting our reconciliation process. Can you investigate?"
        ),
        "ground_truth": {
            "urgency": "normal",
            "department": "technical",
            "response_keywords": ["webhook", "investigate", "event", "endpoint", "logs"],
        },
    },
]



def get_email_by_id(email_id: str) -> Dict[str, Any]:
    """Return an email dict by its email_id. Raises KeyError if not found."""
    for email in EMAILS:
        if email["email_id"] == email_id:
            return email
    raise KeyError(f"Email {email_id} not found in dataset")


def get_all_email_ids() -> List[str]:
    return [e["email_id"] for e in EMAILS]
