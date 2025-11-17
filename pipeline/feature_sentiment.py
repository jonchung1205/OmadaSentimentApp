import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline as hf_pipeline

# Download tokenizer silently
nltk.download("punkt", quiet=True)

#Feature dictionaries
FEATURE_BUCKETS = {
    "AI/Food Logging": [
        "log", "logging", "logged", "track", "tracking", "tracked", "record", "recording",
        "enter", "entering", "input", "add food", "adding food",
        "scan", "scanner", "scanning", "barcode", "bar code",
        "photo", "picture", "image", "camera", "snap", "take a picture",
        "recognize", "recognition", "detect", "detects", "identifying",
        "ai", "artificial intelligence", "machine learning", "smart",
        "automatic", "automatically", "suggests food", "food suggestion",
        "database", "food search", "search for food", "find food",
        "food library", "food list", "can't find", "missing food",
        "limited options", "extensive database",
        "calorie", "calories", "cal", "nutrition", "nutritional", "nutrient", "nutrients",
        "macro", "macros", "protein", "carb", "carbs", "fat", "fiber",
        "portion", "serving", "serving size",
        "recipe", "recipes", "meal", "meals", "breakfast", "lunch", "dinner", "snack",
        "meal prep", "what i ate", "food diary", "food journal",
        "easy to log", "hard to log", "tedious", "time consuming to log",
        "quick entry", "convenient logging"
    ],

    "Engagement/Coach Support": [
        "coach", "coaches", "dietitian", "nutritionist", "trainer",
        "mentor", "advisor", "counselor", "specialist",
        "support", "support team", "customer service", "help desk",
        "human", "real person", "actual person", "live person",
        "expert", "professional", "staff", "team",
        "check in", "check-in", "checking in", "follow up", "following up",
        "message", "messaging", "messages", "text", "texting", "texts",
        "chat", "chatting", "conversation", "talk", "talking",
        "respond", "response", "replies", "reply", "answered",
        "accountability", "accountable", "keep me on track",
        "motivation", "motivate", "motivating", "motivational", "motivated",
        "encouragement", "encouraging", "supportive", "caring",
        "advice", "guidance", "tips", "suggestions", "recommended",
        "feedback", "helpful", "knowledgeable",
        "reminder", "reminders", "daily message", "weekly check",
        "available", "accessible", "responsive", "quick to respond",
        "slow to respond", "never responded", "didn't hear back",
        "community", "forum", "group", "groups", "discussion",
        "other users", "members", "connect with others",
        "someone to talk to", "1-on-1", "one on one", "one-on-one",
        "assigned coach", "my coach", "the coach", "my dietitian"
    ],

    "Personalization": [
        "custom", "customized", "customize", "customizable",
        "personal", "personalized", "personalize",
        "tailored", "tailor", "adapted", "adaptive", "adjust", "adjusted",
        "specific to me", "for me", "fit my needs", "fits my lifestyle",
        "my situation", "individual", "individualized", "unique",
        "goal", "goals", "objective", "objectives", "target", "targets",
        "weight goal", "fitness goal", "health goal", "lose weight",
        "gain weight", "maintain weight", "tone", "build muscle",
        "plan", "meal plan", "eating plan", "diet plan", "nutrition plan",
        "program", "routine", "schedule", "daily plan",
        "suggested meals", "meal suggestions",
        "recommendation", "recommendations", "recommend", "recommends",
        "suggestion", "suggestions", "suggest", "suggests", "suggested",
        "advice", "tips for me",
        "preference", "preferences", "dietary restrictions", "allergies",
        "vegetarian", "vegan", "gluten free", "dairy free",
        "habit", "habits", "lifestyle", "routine",
        "picky eater", "don't like", "dislikes",
        "flexible", "flexibility", "rigid", "strict", "restrictive",
        "one size fits all", "generic", "same for everyone",
        "works for me", "doesn't fit", "not right for me",
        "takes into account", "considers my", "based on my"
    ],

    "Cost/Value": [
        "price", "pricing", "priced", "cost", "costs", "expensive", "pricey",
        "cheap", "affordable", "reasonably priced", "reasonable",
        "value", "worth", "worth it", "not worth", "good value", "poor value",
        "overpriced", "too expensive", "too much", "too costly",
        "great deal", "good deal", "steal", "bargain",
        "subscription", "subscribe", "plan", "membership",
        "monthly", "annual", "yearly", "quarterly",
        "tier", "package", "premium", "basic plan", "pro plan",
        "fee", "fees", "paid", "pay", "paying", "charge", "charged", "charges",
        "money", "dollar", "bucks", "$", "payment", "billing",
        "free", "free version", "free trial", "trial", "trial period",
        "upgrade", "upgraded", "downgrade",
        "premium features", "paywall", "locked", "requires payment",
        "refund", "refunded", "cancel", "cancelled", "canceling", "cancellation",
        "renew", "renewal", "auto-renew", "charged again",
        "compared to", "vs", "versus", "other apps", "competitors",
        "alternatives", "cheaper alternative",
        "wish it was free", "would pay for", "not willing to pay",
        "waste of money", "best investment", "can't afford",
        "hidden costs", "hidden fees", "surprise charge"
    ],

    "Usability/Technical": [
        "bug", "bugs", "buggy", "glitch", "glitches", "glitchy",
        "error", "errors", "problem", "problems", "issue", "issues",
        "broken", "doesn't work", "not working", "won't work", "stopped working",
        "slow", "lag", "laggy", "lagging", "freeze", "freezes", "frozen",
        "crash", "crashes", "crashed", "crashing",
        "hang", "hangs", "stuck", "unresponsive",
        "load", "loading", "loads slowly", "won't load", "takes forever",
        "update", "updated", "updating", "new version", "latest version",
        "since the update", "after update", "broke after",
        "interface", "ui", "ux", "design", "layout", "look", "appearance",
        "menu", "navigation", "navigate", "find things",
        "cluttered", "clean", "organized", "messy", "confusing layout",
        "user friendly", "easy to use", "intuitive", "simple",
        "difficult", "hard to use", "complicated", "confusing",
        "learning curve", "figure out", "not obvious",
        "straightforward", "clear", "unclear",
        "works", "working", "functional", "functions",
        "fix", "fixed", "needs fixing", "repair",
        "sync", "syncing", "syncs", "synchronize",
        "connection", "connect", "disconnect", "offline",
        "bluetooth", "wifi", "internet", "network",
        "login", "log in", "sign in", "password", "username",
        "account", "profile", "forgot password", "locked out",
        "can't access", "won't let me",
        "android", "ios", "iphone", "samsung", "phone",
        "tablet", "ipad", "device", "watch", "apple watch", "fitbit",
        "keeps crashing", "always freezing", "so buggy",
        "clunky", "smooth", "seamless", "frustrating",
        "wish it was easier", "hard to navigate"
    ]
}

#Clause level splitting
def _split_clauses(text: str):
    if not isinstance(text, str):
        return []
    parts = re.split(
        r'\b(?:but|however|although|though|while|and yet|whereas|nevertheless)\b',
        text,
        flags=re.IGNORECASE
    )
    return [p.strip() for p in parts if len(p.strip()) > 3]


#Assign Feature Bucket
def _assign_bucket(text: str):
    t = text.lower()
    for bucket, keywords in FEATURE_BUCKETS.items():
        if any(k in t for k in keywords):
            return bucket
    return None

#Analysis Function
def analyze_sentiment(df: pd.DataFrame, text_column: str = "cleaned_content"):
    model = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    results = []

    for review in df[text_column].dropna():
        for sentence in sent_tokenize(review):
            for clause in _split_clauses(sentence):
                bucket = _assign_bucket(clause)
                if bucket:
                    output = model(clause)[0]
                    results.append({
                        "clause": clause,
                        "feature": bucket,
                        "sentiment": output["label"],
                        "confidence": float(output["score"]),
                        "review": review
                    })

    sent_df = pd.DataFrame(results)

    if sent_df.empty:
        return sent_df, pd.DataFrame()

    summary = (
        sent_df.groupby(["feature", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .assign(total=lambda x: x.sum(axis=1))
    )

    summary["positive_rate"] = summary["POSITIVE"] / summary["total"]
    summary["negative_rate"] = summary["NEGATIVE"] / summary["total"]
    summary["sentiment_score"] = summary["positive_rate"] - summary["negative_rate"]
    summary = summary.sort_values("sentiment_score", ascending=False)

    return sent_df, summary
