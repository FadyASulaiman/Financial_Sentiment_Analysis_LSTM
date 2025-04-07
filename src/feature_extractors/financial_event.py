from collections import Counter
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.feature_extractors.extractor_base import FeatureExtractorBase

class FinancialEventClassifier(FeatureExtractorBase):
    """
    A context-aware feature extractor for classifying financial news sentences into predefined event categories.
    """
    
    EVENT_CATEGORIES = [
        "merger_acquisition",   # Companies merging or one company acquiring another
        "earnings",             # Financial results, profit/loss announcements
        "dividend",             # Dividend payments, changes, suspensions
        "product_launch",       # New product or service releases
        "investment",           # Funding, capital raises, investments
        "restructuring",        # Reorganization, cost-cutting, streamlining operations
        "litigation",           # Lawsuits, legal disputes, settlements
        "executive_change",     # Leadership appointments, resignations, board changes
        "expansion",            # Geographic or business line expansions
        "layoff",               # Staff reductions, job cuts, downsizing
        "partnership",          # Strategic alliances, collaborations, joint ventures
        "regulatory",           # Regulatory approvals, filings, compliance issues
        "stock_movement",       # Share price changes, stock market performance
        "debt_financing",       # Loans, bonds, debt restructuring
        "contract_deal",        # Business contracts, customer agreements, deals
        "other"                 # Catch-all for events not in above categories
    ]
    
    # Keywords associated with each event category for rule-based classification
    EVENT_KEYWORDS = {
        "merger_acquisition": [
            "merger", "acquisition", "acquire", "takeover", "buyout", "consolidation", 
            "combining", "acquiring", "acquired", "merged", "buy ", "buys ", "buying ", 
            "bought ", "acquires ", "merges ", "wins race for", "offers to buy"
        ],
        
        "earnings": [
            "earnings", "profit", "revenue", "financial results", "quarterly", "annual results", 
            "loss", "eps", "ebit", "net sales", "turnover", "forecast", "guidance", 
            "outlook", "ebitda", "income", "profit warning", "revenue growth", "earnings per share",
            "decreased to", "increased to", "rose to", "fell to", "dropped to", "totalled", "totaled",
            "expects to grow", "expected to grow", "expects growth", "estimates", "estimated", "forecast",
            "projected", "profit drop", "profit rise", "profit increase", "profit decrease", "profit fell",
            "profit rose", "results for", "reported", "earnings report", "sales dropped", "sales rose",
            "sales fell", "sales increased", "sales decreased", "profit margin", "profit for the quarter",
            "profit for the year", "full-year"
        ],
        
        "dividend": [
            "dividend", "payout", "distribution", "yield", "shareholder return", "dividend payment",
            "dividend increase", "dividend cut", "dividend suspension", "quarterly dividend",
            "annual dividend", "special dividend", "dividend policy"
        ],
        
        "product_launch": [
            "launch", "introduces", "unveils", "releases", "new product", "new service", "debut", 
            "introducing", "introduced", "announced", "announce", "releasing", "released", "will release",
            "will introduce", "to release", "to introduce", "to launch", "to unveil", "will unveil",
            "product line", "service line", "will be available", "now available", "start deliveries",
            "start the deliveries", "deliveries", "solutions", "solution", "platform", "technology", 
            "product", "service", "flagship"
        ],
        
        "investment": [
            "investment", "funding", "capital", "raised", "invests", "invested", "funding round", 
            "series", "stake", "investor", "venture capital", "invested", "investing", "to invest",
            "will invest", "capital raise", "raise capital", "capital raising", "capital infusion",
            "private equity", "equity investment", "minority stake", "majority stake", "owns", "worth"
        ],
        
        "restructuring": [
            "restructuring", "reorganization", "streamlining", "cost-cutting", "turnaround", 
            "reorganizing", "efficiency", "transformation", "cost reduction", "reorganize", 
            "restructure", "streamline", "optimize", "optimizing", "efficiency improvement",
            "cost savings", "saving costs", "reduce costs", "cutting costs", "operational excellence",
            "operational efficiency", "consolidation", "consolidating", "strategic review", "strategy review"
        ],
        
        "litigation": [
            "lawsuit", "legal", "court", "settlement", "dispute", "litigation", "sue", "sued", 
            "legal action", "case", "judge", "trial", "plaintiff", "defendant", "settlement",
            "judicial", "jurisdiction", "appeal", "appealing", "legal challenge", "legal issue",
            "legal matter", "legal proceeding", "lawsuit", "rebuff", "rebuffed"
        ],
        
        "executive_change": [
            "ceo", "executive", "appoints", "appointed", "resignation", "resigns", "board", 
            "management", "director", "chief", "leadership", "appointed as", "steps down", 
            "promoted to", "new ceo", "new chief", "new executive", "new president", 
            "new chairman", "chief executive", "chief financial", "chief operating", "chief technology",
            "executive director", "managing director", "chairman", "chairwoman", "chairperson",
            "executive committee", "board member", "board of directors", "president", "vice president",
            "head of", "last position", "joins", "join", "joined", "joining", "has appointed", "has named"
        ],
        
        "expansion": [
            "expansion", "enters", "new market", "opens", "facility", "location", "global",
            "growth", "expands", "expanding", "grow", "growing", "expanded", "entry", "entering",
            "establish", "established", "establishing", "open", "opening", "opened", "new office",
            "new location", "new facility", "new plant", "new factory", "geographical expansion",
            "market expansion", "expanding presence", "international expansion", "global expansion",
            "regional expansion", "expand operations", "business expansion", "representative office",
            "sales office"
        ],
        
        "layoff": [
            "layoff", "job cut", "redundancy", "downsizing", "workforce reduction", "staff reduction", 
            "job loss", "terminate", "dismissal", "laid off", "reduces staff", "staff levels", 
            "employment terminated", "termination of employment", "reduce headcount", "headcount reduction",
            "job elimination", "position elimination", "job reduction", "cut jobs", "cutting jobs",
            "reduce jobs", "reducing jobs", "lay off", "lay offs", "layoffs", "staff cuts", "cutting staff",
            "reduce staff", "reducing staff", "downsizing", "downsize", "fire", "fired", "firing"
        ],
        
        "partnership": [
            "partnership", "alliance", "collaboration", "joint venture", "teaming up", "cooperate", 
            "agreement", "partners with", "strategic partnership", "collaborate", "collaborating",
            "partner", "partnering", "cooperative", "cooperation", "jointly", "joint", "signed agreement",
            "strategic alliance", "corporate alliance", "business alliance", "technology partnership",
            "research partnership", "distribution partnership", "marketing partnership", "comarketing",
            "partnership agreement", "partnered with", "partners in", "collaborates with", "working with"
        ],
        
        "regulatory": [
            "regulatory", "regulation", "compliance", "approval", "regulator", "sec", "filing", 
            "approved", "permission", "license", "authorities", "regulatory approval", "regulatory body",
            "regulatory authority", "compliance issue", "regulatory requirement", "regulatory change",
            "regulator", "government approval", "federal approval", "state approval", "agency approval",
            "permit", "permitted", "license", "licensing", "licensed", "regulatory filing", "rule",
            "rules", "standard", "standards", "statutory", "law", "legislation", "legislative",
            "legally", "legal requirements", "regulatory compliance", "authorized", "authorized", "cleared"
        ],
        
        "stock_movement": [
            "stock", "share price", "shares", "trading", "market cap", "valuations", "stockholders", 
            "shareholders", "market value", "exchange", "listed", "ipo", "initial public offering",
            "stock market", "stock exchange", "share", "equity", "securit", "ticker", "traded", 
            "stock closed", "stock opened", "stock rose", "stock fell", "stock dropped", "stock surged",
            "stock plunged", "stock declined", "stock gained", "stock lost", "market capitalization",
            "market valuation", "shares outstanding", "share buyback", "share repurchase", "stock split",
            "reverse split", "stockmarket", "bearish", "bullish", "overbought", "oversold", "downgrade",
            "upgrade", "overweight", "underweight", "price target", "listing", "delisting", "otc"
        ],
        
        "debt_financing": [
            "debt", "loan", "financing", "credit", "bond", "borrowed", "lending", "refinancing", 
            "financial restructuring", "facility", "credit facility", "revolving credit", "term loan",
            "senior debt", "junior debt", "secured debt", "unsecured debt", "high yield", "junk bond",
            "investment grade", "borrowing", "finance", "financed", "financing", "lend", "lending",
            "lender", "borrower", "creditor", "bank loan", "bank debt", "loan agreement", "syndicated loan",
            "credit agreement", "debt issuance", "debt offering", "notes offering", "debt securities",
            "bond issue", "bond offering", "debt restructuring", "debt refinancing", "debt reorganization"
        ],
        
        "contract_deal": [
            "contract", "deal", "agreement", "order", "signed", "client", "customer", "supplier", 
            "vendor", "procurement", "supply", "service contract", "purchase order", "sales contract",
            "purchase contract", "supply agreement", "service agreement", "master agreement", 
            "supply contract", "business contract", "framework agreement", "framework contract",
            "major contract", "significant contract", "important contract", "key contract", "big contract",
            "large contract", "multi-year contract", "multi-million", "deal closing", "agreement finalized",
            "contract awarded", "contract signed", "contract extension", "contract renewal", "closing of",
            "transaction closing", "deal completion", "deal closed", "tender", "bid", "project", "awarded",
            "deal with", "agreement with", "contracted", "deal worth", "contract worth", "value of contract"
        ]
    }
    
    # Context patterns for improved event detection
    CONTEXT_PATTERNS = {
        "merger_acquisition": [
            (r"(acquir\w+|purchas\w+|buy\w*|bought).{1,30}(for|at|worth|valu\w+).{1,15}(\$|€|£|\d+\s*(million|billion|m|b|mln|bln))", 3.0),
            (r"(complet\w+|finaliz\w+|clos\w+).{1,30}(acquisition|takeover|merger|purchase of)", 2.5),
            (r"(enter\w+|sign\w+).{1,30}(agreement|deal).{1,30}(acquir\w+|buy|purchase|tak\w+\s+over)", 2.0),
            (r"(acquire|acquisition of).{1,40}(stake|share|interest|equity).{1,15}(in|of)", 2.0),
            (r"(plan|intend|propose|consider).{1,30}(buy|acquir\w+|tak\w+\s+over)", 1.0)
        ],
        
        "earnings": [
            (r"(report\w+|announce\w+|post\w+|record\w+).{1,30}(quarterly|annual|year|Q\d).{1,30}(result|earnings|profit|revenue|income)", 3.0),
            (r"(increase\w*|decrease\w*|rise|rose|fell|fall\w*|drop\w*).{1,30}(profit|revenue|sales|income|earning).{1,30}(by|to|from).{1,15}(\d+%|\d+\s*%|\$|€|£|\d+)", 2.5),
            (r"(beat|miss\w*|exceed\w*|below).{1,30}(analyst|market|consensus).{1,30}(expectation|estimate|forecast|prediction)", 2.5),
            (r"(guid\w+|forecast|outlook|project\w+).{1,30}(for|of).{1,30}(next|coming|following|full).{1,15}(quarter|year)", 2.0),
            (r"(revenue|sales).{1,30}(of|at|reach\w*).{1,30}(\$|€|£|\d+\s*(million|billion|m|b|mln|bln))", 2.0)
        ],
        
        "dividend": [
            (r"(declar\w+|announce\w+|approv\w+).{1,30}(dividend|distribution).{1,30}(of|at).{1,15}(\$|€|£|\d+|\d+\.\d+|\d+\s*%)", 3.0),
            (r"(increase\w*|decrease\w*|raise|cut|reduc\w+|slash\w+).{1,30}(dividend|payout|distribution).{1,30}(by|to|from).{1,15}(\d+%|\d+\s*%|\$|€|£|\d+)", 2.5),
            (r"(suspend\w*|halt\w*|discontinu\w*|eliminat\w*).{1,30}(dividend|payout|distribution)", 2.5),
            (r"(quarterly|annual|special|one-time).{1,15}(dividend|distribution).{1,30}(of|at).{1,15}(\$|€|£|\d+|\d+\.\d+)", 2.0),
            (r"(dividend|payout).{1,15}(policy|strategy).{1,30}(revis\w+|modif\w+|chang\w+|updat\w+)", 1.5)
        ],
        
        "product_launch": [
            (r"(launch\w*|introduc\w*|unveil\w*|debut\w*|releas\w*).{1,30}(new|novel|innovative).{1,30}(product|service|solution|platform|device|technology)", 3.0),
            (r"(announce\w*).{1,30}(new|latest|next-generation|upgraded).{1,30}(product line|offering|model|version)", 2.5),
            (r"(will|plans to|set to).{1,30}(launch|introduce|release|unveil).{1,30}(new|latest|next|upcoming)", 2.0),
            (r"(present\w*|showcase\w*|demonstrat\w*).{1,30}(new|advanced|cutting-edge|state-of-the-art).{1,30}(technology|product|solution)", 2.0)
        ],
        
        "investment": [
            (r"(invest\w*|fund\w*|financ\w*).{1,30}(\$|€|£|\d+\s*(million|billion|m|b|mln|bln)).{1,30}(in|into|to)", 3.0),
            (r"(secur\w*|rais\w*|obtain\w*).{1,30}(investment|funding|capital|financing).{1,30}(of|worth|valued at).{1,15}(\$|€|£|\d+)", 2.5),
            (r"(complet\w*|clos\w*|finaliz\w*).{1,30}(series|round|phase).{1,15}(funding|investment|financing)", 2.5),
            (r"(acquire\w*|purchase\w*|obtain\w*).{1,30}(stake|share|interest|equity).{1,30}(\d+%|\d+\s*percent)", 2.0)
        ],
        
        "executive_change": [
            (r"(appoint\w*|nam\w*|hire\w*|select\w*).{1,30}(as|to).{1,30}(CEO|Chief|President|Chairman|Director|head)", 3.0),
            (r"(resign\w*|step\w* down|depart\w*|leav\w*).{1,30}(as|from position|from role as).{1,30}(CEO|Chief|President|Chairman|Director)", 3.0),
            (r"(promot\w*|elevat\w*).{1,30}(to|into).{1,30}(role|position).{1,30}(of|as).{1,15}(CEO|Chief|President|Director|head)", 2.5),
            (r"(join\w*).{1,30}(as|to become|to serve as).{1,30}(new|incoming).{1,15}(CEO|Chief|President|Director)", 2.0),
            (r"(board|committee).{1,30}(change|restructur\w*|reorganiz\w*)", 1.5)
        ],
        
        "restructuring": [
            (r"(restructur\w*|reorganiz\w*|transform\w*).{1,30}(business|operation|company|division|department)", 3.0),
            (r"(streamlin\w*|optimi\w*|improv\w*).{1,30}(efficiency|performance|productivity|processes|operations)", 2.5),
            (r"(cost|expense).{1,30}(cut\w*|reduc\w*|sav\w*|control).{1,30}(initiative|program|measure|effort)", 2.5),
            (r"(implement\w*|execut\w*|undergo\w*).{1,30}(restructuring|reorganization|transformation|turnaround).{1,30}(plan|program|initiative)", 2.0),
            (r"(divest\w*|sell\w*|dispos\w*).{1,30}(non-core|underperforming).{1,30}(business|asset|operation|division)", 2.0)
        ],
        
        "litigation": [
            (r"(file\w*|launch\w*|initiate\w*).{1,30}(lawsuit|legal action|legal proceeding|case|claim).{1,30}(against|versus|vs)", 3.0),
            (r"(settle\w*|resolve\w*).{1,30}(lawsuit|litigation|legal dispute|legal case|legal matter).{1,30}(with|for|amount)", 2.5),
            (r"(court|judge|jury).{1,30}(rule\w*|decid\w*|find\w*).{1,30}(in favor|against|liable|not liable)", 2.5),
            (r"(pay|agree\w* to pay).{1,30}(\$|€|£|\d+\s*(million|billion|m|b|mln|bln)).{1,30}(settlement|damages|fine|penalty)", 2.0),
            (r"(appeal\w*|challenge\w*|contest\w*).{1,30}(ruling|decision|judgment|verdict|court order)", 2.0)
        ],
        
        "expansion": [
            (r"(open\w*|launch\w*|establish\w*).{1,30}(new|first).{1,30}(office|store|facility|location|branch|plant|factory).{1,30}(in|at|near)", 3.0),
            (r"(expand\w*|enter\w*|move into).{1,30}(new|additional|foreign|international|global|overseas).{1,30}(market|region|country|territory)", 2.5),
            (r"(growth|expansion).{1,30}(plan|strategy|initiative|effort).{1,30}(for|in|into|across).{1,15}(market|region|segment|sector)", 2.0),
            (r"(increase|expand).{1,30}(presence|footprint|operations).{1,30}(in|across|throughout|within).{1,15}(region|country|market)", 2.0)
        ],
        
        "layoff": [
            (r"(lay off|laid off|cut\w*).{1,30}(\d+|\d+,\d+|\d+\.\d+k|\d+\s*thousand|\d+\s*hundred).{1,30}(employee|staff|worker|job|position)", 3.0),
            (r"(reduc\w*|decreas\w*|slash\w*).{1,30}(workforce|headcount|staff|personnel).{1,30}(by|to).{1,15}(\d+%|\d+\s*percent|\d+|\d+,\d+)", 2.5),
            (r"(announce\w*|plan\w*|implement\w*).{1,30}(layoff|job cut|workforce reduction|redundanc|downsizing).{1,15}(program|plan|measure|initiative)", 2.5),
            (r"(eliminate\w*|cut\w*).{1,30}(\d+|\d+,\d+|\d+\.\d+k|\d+\s*thousand|\d+\s*hundred).{1,30}(position|job|role)", 2.0)
        ],
        
        "partnership": [
            (r"(enter\w*|form\w*|establish\w*|sign\w*).{1,30}(partnership|alliance|collaboration|joint venture).{1,30}(with|agreement)", 3.0),
            (r"(announce\w*|unveil\w*).{1,30}(strategic|new|global).{1,30}(partnership|alliance|collaboration|cooperation).{1,30}(with|between)", 2.5),
            (r"(partner\w*|collaborat\w*|team\w* up|join\w* forces).{1,30}(with|together with).{1,30}(to|for|on).{1,15}(develop|create|provide|deliver|offer)", 2.0),
            (r"(agree\w*|deal|arrangement).{1,30}(to|will).{1,30}(jointly|together|cooperatively).{1,30}(develop|market|sell|distribute)", 2.0)
        ],
        
        "regulatory": [
            (r"(receiv\w*|obtain\w*|secure\w*|grant\w*).{1,30}(regulatory|FDA|SEC|FTC|approval|clearance|authorization|permission).{1,30}(for|to)", 3.0),
            (r"(file\w*|submit\w*|apply\w*).{1,30}(application|request|petition).{1,30}(for|to|with).{1,15}(regulatory|FDA|SEC|FTC|approval)", 2.5),
            (r"(comply\w*|conform\w*|adhere\w*).{1,30}(regulatory|legal|compliance|statutory).{1,30}(requirement|regulation|rule|standard|guideline)", 2.0),
            (r"(investigation|probe|inquiry|review).{1,30}(by|from).{1,30}(regulator|regulatory body|authority|agency|commission)", 2.0)
        ],
        
        "stock_movement": [
            (r"(stock|share).{1,15}(price|value).{1,30}(increase\w*|decrease\w*|rise\w*|fell|drop\w*|surge\w*|plunge\w*|jump\w*).{1,30}(by|to).{1,15}(\d+%|\d+\s*percent|\$|€|£|\d+\.\d+)", 3.0),
            (r"(trading|trade|stock|share).{1,30}(at|reach\w*|hit\w*).{1,30}(all-time|record|new|highest|lowest).{1,15}(high|low|level|price|value)", 2.5),
            (r"(market|stock|share).{1,15}(capitalization|value|worth).{1,30}(\$|€|£|\d+\s*(million|billion|trillion|m|b|t|mln|bln))", 2.0),
            (r"(analyst|investment bank|broker).{1,30}(upgrade\w*|downgrade\w*|raise\w*|lower\w*|adjust\w*).{1,30}(rating|recommendation|price target|outlook)", 2.0)
        ],
        
        "debt_financing": [
            (r"(raise\w*|secure\w*|obtain\w*|close\w*).{1,30}(\$|€|£|\d+\s*(million|billion|m|b|mln|bln)).{1,30}(debt|loan|financing|credit facility|term loan)", 3.0),
            (r"(issue\w*|sell\w*|offer\w*|place\w*).{1,30}(bond|note|debt security|senior note|subordinated note).{1,30}(worth|valued|amount|totaling).{1,15}(\$|€|£|\d+)", 2.5),
            (r"(refinanc\w*|restructur\w*|renegotiat\w*).{1,30}(debt|loan|credit facility|term loan|bond|note).{1,30}(of|worth|amount|totaling).{1,15}(\$|€|£|\d+)", 2.0),
            (r"(enter\w*|sign\w*|secure\w*).{1,30}(credit|loan|debt).{1,15}(agreement|facility|arrangement).{1,30}(with|from).{1,15}(bank|lender|financial institution)", 2.0)
        ],
        
        "contract_deal": [
            (r"(sign\w*|enter\w*|secure\w*|win\w*|award\w*).{1,30}(contract|deal|agreement).{1,30}(worth|valued at|amount|value).{1,15}(\$|€|£|\d+\s*(million|billion|m|b|mln|bln))", 3.0),
            (r"(award\w*|grant\w*|give\w*).{1,30}(\$|€|£|\d+\s*(million|billion|m|b|mln|bln)).{1,30}(contract|deal|agreement|order)", 2.5),
            (r"(multi-year|long-term|major|significant|strategic).{1,15}(contract|agreement|deal).{1,30}(with|to|for).{1,30}(provide|supply|deliver|support)", 2.0),
            (r"(renew\w*|extend\w*|expand\w*).{1,30}(existing|current|ongoing).{1,30}(contract|agreement|deal).{1,30}(with|for|to)", 2.0)
        ]
    }
    
    # Keywords with negative context indicators
    NEGATIVE_CONTEXTS = {
        "merger_acquisition": [
            (r"(not|no longer|doesn't|does not|won't|will not).{1,30}(acquir\w+|buy|purchase)", 3.0),
            (r"(deny|denies|denied|refut\w+|dismiss\w+).{1,30}(rumor|speculation|report).{1,30}(acquisition|merger|takeover)", 3.0),
            (r"(cancel\w+|terminat\w+|abandon\w+|end\w+).{1,30}(acquisition|merger|takeover|deal)", 2.5)
        ],
        
        "earnings": [
            (r"(not|doesn't|does not|won't|will not).{1,30}(report|announce|release).{1,30}(earnings|results)", 3.0),
            (r"(delay\w+|postpon\w+|defer\w+).{1,30}(earnings|results).{1,30}(report|announcement|release)", 2.5)
        ],
        
        "product_launch": [
            (r"(delay\w+|postpon\w+|cancel\w+|suspend\w+).{1,30}(launch|release|introduction|debut).{1,30}(of|for).{1,15}(product|service)", 3.0),
            (r"(not|won't|will not).{1,30}(launch|release|introduce|unveil).{1,30}(product|service|solution)", 2.5)
        ],
        
        "executive_change": [
            (r"(deny|denies|denied|refut\w+|dismiss\w+).{1,30}(rumor|speculation|report).{1,30}(resign|departure|leaving|stepping down)", 3.0),
            (r"(not|no|won't|will not).{1,30}(step down|resign|leave|depart)", 2.5)
        ],
        
        "partnership": [
            (r"(end\w*|terminate\w*|dissolve\w*|cancel\w*).{1,30}(partnership|alliance|collaboration|joint venture)", 3.0),
            (r"(not|no longer).{1,30}(partner|collaborate|cooperate|work together).{1,30}(with|on)", 2.5)
        ]
    }
    
    # Entity type patterns
    ENTITY_PATTERNS = {
        "company": r"([A-Z][a-z]*\.?\s)?([A-Z][a-z]+\s)*[A-Z][a-z]*\.?(\s(Inc|Corp|Co|Ltd|LLC|Group|SA|AG|SE|NV|PLC|GmbH)\.?)?",
        "money": r"(\$|€|£|USD|EUR|GBP)?\s?\d+(\.\d+)?\s?(million|billion|trillion|mn|bn|tn|m|b|t)?(\s(USD|EUR|GBP))?",
        "percentage": r"\d+(\.\d+)?\s?(%|percent|pct|basis points|bps)",
        "date": r"(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\s\d{1,2}(st|nd|rd|th)?(,?\s\d{4})?",
    }
    
    # Key financial verbs with sentiment
    FINANCIAL_VERBS = {
        "positive": ["increase", "rise", "grow", "improve", "expand", "strengthen", "accelerate", "exceed", "beat", "outperform", "gain"],
        "negative": ["decrease", "decline", "drop", "fall", "reduce", "weaken", "slow", "miss", "underperform", "lose"],
        "neutral": ["report", "announce", "state", "declare", "disclose", "release", "publish", "issue", "present"]
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize the FinancialEventClassifier with context-aware pattern matching.
        
        Args:
            config (Dict): Configuration dictionary with the following possible keys:
                - input_col (str): Name of the input column containing text (default: 'Sentence')
                - output_col (str): Name of the output column for event classification (default: 'Event')
                - preprocess (bool): Whether to preprocess text before classification (default: True)
                - add_confidence (bool): Whether to add confidence scores as additional columns (default: False)
                - min_confidence (float): Minimum confidence threshold to classify (default: 0.3)
                - use_contextual (bool): Whether to use contextual pattern matching (default: True)
                - context_weight (float): Weight for contextual pattern match scores (default: 1.5)
        """
        # Initialize all patterns with compiled regex
        self._compile_patterns()
        
        super().__init__(config)
        
        # Set default configuration values if not provided
        self.input_col = self.config.get('input_col', 'Sentence')
        self.output_col = self.config.get('output_col', 'Event')
        self.preprocess = self.config.get('preprocess', True)
        self.add_confidence = self.config.get('add_confidence', False)
        self.min_confidence = self.config.get('min_confidence', 0.2)
        self.use_contextual = self.config.get('use_contextual', True)
        self.context_weight = self.config.get('context_weight', 1.4)
        
        # Initialize NLP components
        self._init_nlp()
    
    def _compile_patterns(self):
        """Compile regex patterns for faster matching"""
        # Compile context patterns
        for category in self.CONTEXT_PATTERNS:
            compiled_patterns = []
            for pattern, weight in self.CONTEXT_PATTERNS[category]:
                compiled_patterns.append((re.compile(pattern, re.IGNORECASE), weight))
            self.CONTEXT_PATTERNS[category] = compiled_patterns
            
        # Compile negative contexts
        for category in self.NEGATIVE_CONTEXTS:
            compiled_patterns = []
            for pattern, weight in self.NEGATIVE_CONTEXTS[category]:
                compiled_patterns.append((re.compile(pattern, re.IGNORECASE), weight))
            self.NEGATIVE_CONTEXTS[category] = compiled_patterns
            
        # Compile entity patterns
        for entity_type in self.ENTITY_PATTERNS:
            self.ENTITY_PATTERNS[entity_type] = re.compile(self.ENTITY_PATTERNS[entity_type])
    
    def _init_nlp(self):
        """Initialize NLP components"""
        if self.preprocess:
            # Download necessary NLTK resources
            try:
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('stopwords')
                nltk.download('wordnet')
                nltk.download('punkt')
                
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Regular expressions for detecting financial patterns
            self.money_pattern = re.compile(r'(\$|€|£|USD|EUR|GBP|million|billion|mn|bn|\d+\.\d+|\d+%)')
            self.ticker_pattern = re.compile(r'\$[A-Z]{1,5}')
            
            # Add named entity recognition patterns
            self.companies_pattern = re.compile(r'([A-Z][a-z]*\.?\s)?([A-Z][a-z]+\s)*[A-Z][a-z]*\.?(\s(Inc|Corp|Co|Ltd|LLC|Group|SA|AG|SE|NV|PLC|GmbH)\.?)?')
    
    def preprocess_text(self, text):
        """
        Preprocess text for classification with more advanced techniques.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        if not self.preprocess:
            return text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Replace ticker symbols with a token
        text = self.ticker_pattern.sub(" TICKERSYMBOL ", text)
        
        # Replace monetary values with a token but preserve the value
        text = self.money_pattern.sub(lambda m: f" MONEYVALUE_{m.group(0).replace(' ', '_')} ", text)
        
        # Identify company names
        text = self.companies_pattern.sub(" COMPANY ", text)
        
        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return " ".join(tokens)
        
    def find_entities(self, text):
        """
        Extract financial entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of entity types and their matches
        """
        entities = {}
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = matches
                
        return entities
    
    def extract_sentence_structure(self, text):
        """
        Extract basic sentence structure features.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary with sentence features
        """
        # Tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Get parts of speech with a basic approach
        nouns = []
        verbs = []
        
        for token in tokens:
            if token.isalpha() and len(token) > 2:
                # This is a very basic approximation - ideally use proper POS tagging
                if token.endswith('ing') or token.endswith('ed') or token.endswith('s'):
                    verbs.append(token)
                else:
                    nouns.append(token)
        
        # Identify verb sentiment
        verb_sentiment = "neutral"
        for verb in verbs:
            verb_stem = self.lemmatizer.lemmatize(verb, 'v')
            if verb_stem in self.FINANCIAL_VERBS["positive"]:
                verb_sentiment = "positive"
                break
            elif verb_stem in self.FINANCIAL_VERBS["negative"]:
                verb_sentiment = "negative"
                break
        
        return {
            "tokens": tokens,
            "nouns": nouns,
            "verbs": verbs,
            "verb_sentiment": verb_sentiment,
            "token_count": len(tokens)
        }
    
    def detect_contextual_patterns(self, text):
        """
        Detect contextual patterns in the text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of category matches and their scores
        """
        if not self.use_contextual:
            return {}
            
        scores = {}
        matches = {}
        
        # Check contextual patterns for each category
        for category in self.CONTEXT_PATTERNS:
            category_score = 0
            category_matches = []
            
            # Try each pattern
            for pattern, weight in self.CONTEXT_PATTERNS[category]:
                pattern_matches = pattern.findall(text.lower())
                if pattern_matches:
                    category_score += weight * len(pattern_matches)
                    category_matches.extend(pattern_matches)
            
            if category_score > 0:
                scores[category] = category_score
                matches[category] = category_matches
                
        # Check negative contexts
        for category in self.NEGATIVE_CONTEXTS:
            if category in scores:
                for pattern, weight in self.NEGATIVE_CONTEXTS[category]:
                    neg_matches = pattern.findall(text.lower())
                    if neg_matches:
                        # Reduce or negate the score based on negative context
                        scores[category] -= weight * len(neg_matches)
                        
                # Remove if score is negative after accounting for negative context
                if scores[category] <= 0:
                    del scores[category]
                    del matches[category]
        
        return {
            "scores": scores,
            "matches": matches
        }
    
    def classify_with_keywords(self, text):
        """
        Classify text using keyword matching from the original approach.
        
        Args:
            text (str): Text to classify
            
        Returns:
            tuple: (category_scores, matched_keywords)
        """
        if not isinstance(text, str):
            return {}, {}
            
        text = text.lower()
        scores = {}
        matched_keywords = {}
        
        # Calculate scores for each category based on keyword matches
        for category, keywords in self.EVENT_KEYWORDS.items():
            category_score = 0
            matches = []
            
            for keyword in keywords:
                if keyword.lower() in text:
                    # Weight multi-word keywords higher
                    if " " in keyword:
                        category_score += 1.5
                    else:
                        category_score += 1.0
                    matches.append(keyword)
            
            if category_score > 0:
                # Normalize by the number of keywords
                scores[category] = category_score / len(keywords)
                matched_keywords[category] = matches
        
        return scores, matched_keywords
    
    def classify_text(self, text):
        """
        Classify text using combined keyword and contextual pattern matching.
        
        Args:
            text (str): Text to classify
            
        Returns:
            tuple: (predicted_category, confidence, all_scores)
        """
        if not isinstance(text, str):
            return "other", 1.0, {"other": 1.0}
            
        # Preprocess text
        preprocessed = text.lower()
        
        # Get scores from keyword matching
        keyword_scores, matched_keywords = self.classify_with_keywords(preprocessed)
        
        # Get scores from contextual pattern matching
        context_results = self.detect_contextual_patterns(text)
        context_scores = context_results.get("scores", {})
        
        # Extract entities and sentence structure for additional features
        entities = self.find_entities(text)
        sentence_features = self.extract_sentence_structure(text)
        
        # Combine scores
        combined_scores = {}
        
        # Add all categories from both methods
        all_categories = set(list(keyword_scores.keys()) + list(context_scores.keys()))
        
        for category in all_categories:
            keyword_score = keyword_scores.get(category, 0)
            context_score = context_scores.get(category, 0)
            
            # Weight contextual patterns higher
            if self.use_contextual:
                combined_scores[category] = keyword_score + (context_score * self.context_weight)
            else:
                combined_scores[category] = keyword_score
        
        # Apply additional heuristics
        
        # 1. If we have money entities and financial verbs, boost certain categories
        if 'money' in entities and sentence_features['verbs']:
            if sentence_features['verb_sentiment'] == 'positive':
                # Boost positive financial events
                for category in ['earnings', 'investment', 'expansion']:
                    if category in combined_scores:
                        combined_scores[category] *= 1.3
            elif sentence_features['verb_sentiment'] == 'negative':
                # Boost negative financial events
                for category in ['layoff', 'restructuring', 'debt_financing']:
                    if category in combined_scores:
                        combined_scores[category] *= 1.3
        
        # 2. For very short messages, require higher confidence
        if sentence_features['token_count'] < 8:
            for category in combined_scores:
                combined_scores[category] *= 0.8
        
        # If no category matched, return "other"
        if not combined_scores:
            return "other", 1.0, {"other": 1.0}
        
        # Normalize scores
        total_score = sum(combined_scores.values())
        for category in combined_scores:
            combined_scores[category] /= total_score
        
        # Get the category with the highest score
        predicted_category = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[predicted_category]
        
        # Apply minimum confidence threshold
        if confidence < self.min_confidence:
            return "other", confidence, combined_scores
        
        return predicted_category, confidence, combined_scores
    
    def fit(self, X, y=None):
        """
        No training needed for rule-based approach, but method required for scikit-learn compatibility.
        
        Args:
            X: Input data (ignored)
            y: Target labels (ignored)
            
        Returns:
            self: The classifier instance
        """
        # Nothing to do for rule-based approach
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding event classification.
        
        Args:
            X (pd.DataFrame or pd.Series or list): Input data containing text samples
            
        Returns:
            pd.DataFrame: Transformed data with event classification
        """
        # Copy the input data to avoid modifying the original
        if isinstance(X, pd.DataFrame):
            result = X.copy()
            X_text = X[self.input_col]
        else:
            # If X is a Series or list, convert to DataFrame
            if isinstance(X, pd.Series):
                X_text = X
                result = pd.DataFrame({self.input_col: X})
            else:
                X_text = X
                result = pd.DataFrame({self.input_col: X})
        
        # Convert to list if Series
        if isinstance(X_text, pd.Series):
            X_text = X_text.tolist()
        
        # Get event predictions
        event_predictions = []
        confidence_scores = []
        category_scores = []
        
        for text in X_text:
            category, confidence, scores = self.classify_text(text)
            event_predictions.append(category)
            confidence_scores.append(confidence)
            category_scores.append(scores)
        
        # Add the predictions as a new column
        result[self.output_col] = event_predictions
        
        # Optionally add confidence scores
        if self.add_confidence:
            result[f"{self.output_col}_confidence"] = confidence_scores
            
            # Optionally add individual category scores
            if self.config.get('add_category_scores', False):
                for category in self.EVENT_CATEGORIES:
                    result[f"{category}_score"] = [scores.get(category, 0.0) for scores in category_scores]
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """
        Return the names of features produced by this feature extractor.
        
        Returns:
            List[str]: List of feature names
        """
        feature_names = [self.output_col]
        
        if self.add_confidence:
            feature_names.append(f"{self.output_col}_confidence")
            
            if self.config.get('add_category_scores', False):
                for category in self.EVENT_CATEGORIES:
                    feature_names.append(f"{category}_score")
                    
        return feature_names
    
    def predict(self, texts):
        """
        Predict the event category for input texts.
        
        Args:
            texts (list or str or pd.Series): Input text(s) to classify
            
        Returns:
            list: Predicted event categories
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform the texts and extract the predictions
        result = self.transform(texts)
        
        return result[self.output_col].tolist()
    
    def explain_classification(self, text):
        """
        Explain why a text was classified in a particular way.
        
        Args:
            text (str): Text to explain
            
        Returns:
            dict: Explanation of the classification
        """
        # Preprocess text
        preprocessed = text.lower()
        
        # Get scores and matched keywords
        keyword_scores, matched_keywords = self.classify_with_keywords(preprocessed)
        
        # Get context patterns
        context_results = self.detect_contextual_patterns(text)
        context_scores = context_results.get("scores", {})
        context_matches = context_results.get("matches", {})
        
        # Get entities
        entities = self.find_entities(text)
        
        # Get sentence structure
        sentence_features = self.extract_sentence_structure(text)
        
        # Get final classification
        category, confidence, all_scores = self.classify_text(text)
        
        # Build explanation
        explanation = {
            "classification": category,
            "confidence": confidence,
            "all_category_scores": all_scores,
            "keyword_evidence": matched_keywords,
            "contextual_evidence": context_matches,
            "entities_detected": entities,
            "sentence_features": {
                "verbs": sentence_features["verbs"],
                "verb_sentiment": sentence_features["verb_sentiment"],
                "length": sentence_features["token_count"]
            }
        }
        
        return explanation
    
    def update_keywords(self, category, new_keywords, replace=False):
        """
        Update the keywords for a specific category.
        
        Args:
            category (str): Category to update
            new_keywords (list): New keywords to add
            replace (bool): Whether to replace existing keywords or append (default: False)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if category not in self.EVENT_CATEGORIES:
            print(f"Error: Category '{category}' not found")
            return False
            
        if replace:
            self.EVENT_KEYWORDS[category] = new_keywords
        else:
            # Add only unique keywords
            existing_keywords = set(self.EVENT_KEYWORDS[category])
            for keyword in new_keywords:
                if keyword not in existing_keywords:
                    self.EVENT_KEYWORDS[category].append(keyword)
        
        return True
    
    def add_context_pattern(self, category, pattern, weight=1.0):
        """
        Add a new context pattern for a category.
        
        Args:
            category (str): Category to add pattern for
            pattern (str): Regular expression pattern
            weight (float): Weight for the pattern
            
        Returns:
            bool: True if successful, False otherwise
        """
        if category not in self.EVENT_CATEGORIES:
            print(f"Error: Category '{category}' not found")
            return False
        
        # Compile pattern
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            print(f"Error: Invalid regex pattern '{pattern}'")
            return False
        
        # Initialize pattern list if needed
        if category not in self.CONTEXT_PATTERNS:
            self.CONTEXT_PATTERNS[category] = []
        
        # Add pattern
        self.CONTEXT_PATTERNS[category].append((compiled_pattern, weight))
        
        return True
    
    def analyze_unclassified(self, texts, threshold=0.3):
        """
        Analyze sentences that fall below the confidence threshold.
        
        Args:
            texts (list): List of texts to analyze
            threshold (float): Confidence threshold
            
        Returns:
            dict: Analysis of unclassified texts
        """
        unclassified = []
        low_confidence = []
        category_distribution = {}
        
        for text in texts:
            category, confidence, scores = self.classify_text(text)
            
            if confidence < threshold:
                unclassified.append({
                    "text": text,
                    "best_category": category,
                    "confidence": confidence,
                    "top_scores": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                })
            elif confidence < 0.5:
                low_confidence.append({
                    "text": text,
                    "category": category,
                    "confidence": confidence
                })
            
            # Track category distribution
            if category not in category_distribution:
                category_distribution[category] = 0
            category_distribution[category] += 1
        
        # Calculate percentage of unclassified
        unclassified_pct = (len(unclassified) / len(texts)) * 100 if texts else 0
        
        return {
            "unclassified_count": len(unclassified),
            "unclassified_percentage": unclassified_pct,
            "low_confidence_count": len(low_confidence),
            "category_distribution": category_distribution,
            "unclassified_examples": unclassified[:10],  # First 10 examples
            "common_patterns": self._extract_common_patterns(unclassified)
        }
    
    def _extract_common_patterns(self, unclassified_texts):
        """Extract common patterns from unclassified texts"""
        if not unclassified_texts:
            return []
            
        # Extract words that appear frequently
        all_words = []
        for item in unclassified_texts:
            tokens = nltk.word_tokenize(item["text"].lower())
            all_words.extend([t for t in tokens if t.isalpha() and t not in self.stop_words])
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Get most common words
        common_words = word_counts.most_common(20)
        
        # Look for bigrams
        bigrams = []
        for item in unclassified_texts:
            tokens = nltk.word_tokenize(item["text"].lower())
            for i in range(len(tokens) - 1):
                if tokens[i].isalpha() and tokens[i+1].isalpha():
                    bigrams.append((tokens[i], tokens[i+1]))
        
        # Count bigram frequencies
        bigram_counts = Counter(bigrams)
        
        # Get most common bigrams
        common_bigrams = bigram_counts.most_common(10)
        
        return {
            "common_words": common_words,
            "common_bigrams": common_bigrams
        }
    
    def self_improve(self, texts, min_confidence=0.7):
        """
        Self-improve by analyzing high-confidence classifications.
        
        Args:
            texts (list): List of texts to analyze
            min_confidence (float): Minimum confidence to consider
            
        Returns:
            dict: Results of self-improvement
        """
        high_confidence_samples = {}
        
        # Collect high confidence samples for each category
        for text in texts:
            category, confidence, _ = self.classify_text(text)
            
            if confidence >= min_confidence:
                if category not in high_confidence_samples:
                    high_confidence_samples[category] = []
                    
                high_confidence_samples[category].append((text, confidence))
        
        # Use these to improve keyword lists
        improvements = {}
        
        for category, samples in high_confidence_samples.items():
            if len(samples) < 5:  # Need enough samples
                continue
                
            # Extract potentially useful new keywords
            all_text = " ".join([self.preprocess_text(s[0]) for s in samples])
            words = all_text.split()
            word_counts = Counter(words)
            
            # Find words that might be good indicators but aren't in keywords
            existing_keywords = set([kw.lower() for kw in self.EVENT_KEYWORDS.get(category, [])])
            
            new_keywords = []
            for word, count in word_counts.most_common(50):
                # Filter to keep only good potential keywords
                if (len(word) > 3 and 
                    word not in self.stop_words and 
                    word.lower() not in existing_keywords and
                    count >= 3):  # Appears in multiple samples
                    new_keywords.append(word)
            
            # Add the best new keywords
            if new_keywords:
                added = self.update_keywords(category, new_keywords[:5])
                improvements[category] = new_keywords[:5]
        
        return {
            "categories_improved": list(improvements.keys()),
            "new_keywords_added": improvements,
            "high_confidence_samples_found": {k: len(v) for k, v in high_confidence_samples.items()}
        }