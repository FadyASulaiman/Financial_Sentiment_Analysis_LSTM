class FinancialLexicon:
    '''Financial lexicons and gazetteers for feature engineering'''
    
    # Financial sentiment lexicon
    POSITIVE_TERMS = {
        'beat', 'beats', 'exceeded', 'exceeds', 'outperform', 'outperforms', 'outperformed',
        'bullish', 'growth', 'growing', 'grew', 'increase', 'increased', 'increases',
        'higher', 'up', 'upward', 'rise', 'rising', 'rose', 'gain', 'gains', 'gained',
        'positive', 'profit', 'profitable', 'profitability', 'strong', 'stronger',
        'strength', 'success', 'successful', 'improve', 'improved', 'improvement',
        'rally', 'rallied', 'rallies', 'record', 'recovery', 'efficient', 'opportunity',
        'opportunities', 'advantage', 'advantages', 'promising', 'prospect', 'prospects',
        'surge', 'surged', 'surges', 'jump', 'jumped', 'jumps', 'soar', 'soared', 'soars'
    }
    
    NEGATIVE_TERMS = {
        'miss', 'missed', 'misses', 'underperform', 'underperforms', 'underperformed',
        'bearish', 'decline', 'declined', 'declines', 'decrease', 'decreased', 'decreases',
        'lower', 'down', 'downward', 'fall', 'falling', 'fell', 'loss', 'losses', 'lost',
        'negative', 'weak', 'weaker', 'weakness', 'fail', 'failed', 'failure', 'poor',
        'worsen', 'worsened', 'worsening', 'concern', 'concerned', 'concerns', 'difficult',
        'difficulty', 'struggle', 'struggled', 'struggles', 'challenging', 'challenge',
        'challenges', 'problem', 'problems', 'risk', 'risks', 'risky', 'uncertain',
        'uncertainty', 'volatility', 'volatile', 'pressure', 'pressured', 'drop', 'dropped',
        'drops', 'plunge', 'plunged', 'plunges', 'tumble', 'tumbled', 'tumbles', 'slump',
        'slumped', 'slumps', 'crash', 'crashed', 'crashes'
    }
    
    # Companies gazetteer (top 100 global companies)
    COMPANIES = {
        'apple', 'microsoft', 'amazon', 'alphabet', 'google', 'facebook', 'meta',
        'tesla', 'berkshire', 'hathaway', 'jpmorgan', 'chase', 'visa', 'walmart',
        'johnson', 'procter', 'gamble', 'mastercard', 'bank', 'america', 'disney',
        'verizon', 'coca', 'cola', 'netflix', 'pfizer', 'intel', 'cisco', 'adobe',
        'pepsi', 'salesforce', 'bofa', 'citigroup', 'wells', 'fargo', 'toyota',
        'samsung', 'shell', 'bp', 'exxon', 'mobil', 'chevron', 'total', 'siemens',
        'nokia', 'ericsson', 'ibm', 'oracle', 'sap', 'huawei', 'sony', 'nintendo',
        'goldman', 'sachs', 'morgan', 'stanley', 'blackrock', 'hsbc', 'barclays',
        'deutsche', 'ubs', 'credit', 'suisse', 'amd', 'nvidia', 'qualcomm',
        'twitter', 'snapchat', 'linkedin', 'uber', 'lyft', 'airbnb', 'tesla',
        'nokia', 'ericsson', 'huhtamaki', 'stockmann', 'fiskars', 'benefon'
    }
    
    # Financial market indicators
    MARKET_INDICATORS = {
        'gdp', 'inflation', 'cpi', 'ppi', 'unemployment', 'interest', 'rate',
        'fomc', 'fed', 'ecb', 'boj', 'pmi', 'ism', 'retail', 'sales', 'housing',
        'starts', 'sentiment', 'confidence', 'index', 'yield', 'bond', 'treasury',
        'deficit', 'surplus', 'debt', 'earnings', 'revenue', 'eps', 'p/e', 'ratio'
    }
    

    # Define the 15 most common financial events with clear definitions
    FINANCIAL_EVENTS = [
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
        "contract_deal"         # Business contracts, customer agreements, deals
    ]


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
        "date": r"(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\s\d{1,2}(st|nd|rd|th)?(,?\s\d{4})?"
    }

    # Key financial verbs with sentiment
    FINANCIAL_VERBS = {
        "positive": ["increase", "rise", "grow", "improve", "expand", "strengthen", "accelerate", "exceed", "beat", "outperform", "gain"],
        "negative": ["decrease", "decline", "drop", "fall", "reduce", "weaken", "slow", "miss", "underperform", "lose"],
        "neutral": ["report", "announce", "state", "declare", "disclose", "release", "publish", "issue", "present"]
    }
    

    FINANCIAL_EVENT_PATTERNS = {
            "merger_acquisition": [
                r"\b(merger|acquisition|takeover|acquire[ds]?|merged?|bid for)\b",
                r"\b(purchas(e[ds]?|ing)|bought|buy|buys) .{1,20}\b(company|stake|share|business|firm)\b",
                r"\btake[ns]? (over|control of)\b",
                r"\b((\d{1,3})(\.\d+)?\s?(%|percent)\s?stake)\b",
                r"\b(alliance plans)\b"
            ],
            
            "earnings": [
                r"\b(earnings|revenue|profit|loss|income|EBITDA|EPS|sales) (grew|rose|increased|decreased|fell|dropped|reached|totaled|amounted to)\b",
                r"\b(financial|quarterly|annual|first[- ]quarter|second[- ]quarter|third[- ]quarter|fourth[- ]quarter|Q[1-4]|([1-4]Q)) results\b",
                r"\b(report(ed|ing)|post(ed|ing)|announce[ds]?) .{1,30}\b(profit|loss|revenue|earnings)\b",
                r"\b(operat(ing|ed)) (profit|loss|margin)\b",
                r"\b(net|gross) (income|profit|loss)\b",
                r"\bfiscal (year|quarter)\b",
                r"\bsales .{1,15} (EUR|USD|€|\$) .{1,15} (million|billion|mn|bn)\b"
            ],
            
            "dividend": [
                r"\bdividend(s)?\b",
                r"\b(declar(e[ds]?|ing)|announce[ds]?|rais(e[ds]?|ing)|increas(e[ds]?|ing)|cut[s]?|reduc(e[ds]?|ing)|suspend(ed|ing)?) dividend\b",
                r"\b(pay(s|ed|ing)?|distribut(e[ds]?|ing)) .{1,20}\b((to|for) (share|stock)holders)\b",
                r"\b(quarterly|annual|special|interim) (dividend|payment|distribution)\b",
                r"\bpayout ratio\b",
                r"\bdividend yield\b"
            ],
            
            "product_launch": [
                r"\b(launch(ed|es|ing)?|introduc(e[ds]?|ing)|unveil(ed|ing)?|releas(e[ds]?|ing)) .{1,30}\b(new|product|service|solution|platform)\b",
                r"\bnew (product|service|solution|platform|technology)\b",
                r"\b(product|range) (portfolio|line|offering)\b",
                r"\b(next[\s-]generation|cutting[\s-]edge) .{1,20}\b(product|solution|technology|platform)\b",
                r"\b(begin|start)(s|ed|ing)? (production|manufacturing)\b",
                r"\bmade with\b"
            ],
            
            "investment": [
                r"\b(invest(s|ed|ing|ment)|funding|capital|raised)\b",
                r"\b(venture|private equity|seed) (capital|investment|funding)\b",
                r"\b(series|round) [A-Z]\b",
                r"\b(invest(s|ed|ing)?|fund(s|ed|ing)?) .{1,30}\b(project|expansion|development)\b",
                r"\binvestment .{1,20}\b(totaling|worth|valued at)\b",
                r"\b(million|billion|mn|bn) .{1,15}\b(investment|funding|capital)\b"
            ],
            
            "restructuring": [
                r"\b(restructur(e[ds]?|ing)|reorganiz(e[ds]?|ing)|transform(s|ed|ing|ation))\b",
                r"\b(cost|operational) (reduction|cutting|saving)(s)?\b",
                r"\b(efficiency|streamlining) (program|measure|initiative)(s)?\b",
                r"\bimprov(e[ds]?|ing) (business|operational) (model|structure)\b",
                r"\b(divest(s|ed|ing|iture)|sell(s|ing)) .{1,20}\b(unit|division|business|subsidiary)\b"
            ],
            
            "litigation": [
                r"\b(litigation|lawsuit|legal action|court case|dispute|sue[ds]?|sued)\b",
                r"\b(legal|court) (proceedings|case|battle|fight|action)\b",
                r"\b(antitrust|regulatory) (investigation|probe|inquiry)\b",
                r"\b(settle[ds]?|settlement|resolution) .{1,20}\b(case|dispute|claim|lawsuit)\b",
                r"\b(fine[ds]?|penalty|damages)\b",
                r"\bin jeopardy\b",
                r"\bsanctions\b"
            ],
            
            "executive_change": [
                r"\b(CEO|CFO|CTO|COO|chief|executive|officer|director|board|chairman) .{1,30}\b(appoint(ed|s)?|resign(ed|s)?|stepped? down|left|joins|joining|named)\b",
                r"\b(appoint(ed|s)?|hir(ed|es|ing)|nam(ed|es|ing)|elect(ed|s)?) .{1,20}\b(as|to) .{1,15}\b(CEO|CFO|CTO|COO|chief|executive|position|director|board)\b",
                r"\b(management|leadership) (change|team|structure)\b",
                r"\b(resign(ed|s|ing)?|depart(ed|s|ing)?|step(s|ped)? down|leaving)\b .{1,20}\b(CEO|CFO|CTO|COO|chief|executive|position)\b",
                r"\b(succeed(s|ed)?|replace(s|d)?) .{1,20}\b(as|by) .{1,15}\b(CEO|CFO|CTO|COO|chief|executive)\b",
                r"\bdirector .{1,20}\b(said)\b"
            ],
            
            "expansion": [
                r"\b(expand(s|ed|ing)?|grow(s|ed|ing)?) .{1,20}\b(into|in|to) .{1,20}\b(market|business|region|country|segment)\b",
                r"\b(new|international|global) (market|facility|location|office|store|region)\b",
                r"\b(open(s|ed|ing)?|establish(es|ed|ing)?|launch(es|ed|ing)?) .{1,20}\b(office|facility|store|presence|subsidiary)\b",
                r"\b(entry|expansion) (into|in) ([A-Z]\w+)\b",
                r"\b(global|international|worldwide) (reach|presence|footprint)\b",
                r"\bgains? foothold\b"
            ],
            
            "layoff": [
                r"\b(layoff(s)?|downsiz(e[ds]?|ing)|job cut(s)?|cut(s|ting)? job(s)?|redundanc(y|ies))\b",
                r"\b(reduc(e[ds]?|ing)|cut(s|ting)?) .{1,20}\b(workforce|staff|employees|personnel|headcount)\b",
                r"\b(eliminate[ds]?|shed(s|ding)?) .{1,20}\b(position(s)?|job(s)?)\b",
                r"\b(employee|worker|staff) (reduction|termination)\b",
                r"\b(lose|lost|loose) .{1,15}\bjobs\b",
                r"\b(lay[- ]offs?|job losses)\b",
                r"\btemporary lay[- ]offs?\b",
                r"\blose .{1,15}\bjobs\b"
            ],
            
            "partnership": [
                r"\b(partnership|collaboration|alliance|joint venture)\b",
                r"\b(partner(s|ed|ing)?|collaborat(e[ds]?|ing)|team(s|ed)? up) with\b",
                r"\b(strategic|key) (relationship|partnership|alliance)\b",
                r"\b(sign(s|ed)?|enter(s|ed)?|form(s|ed)?) .{1,20}\b(agreement|deal|partnership)\b",
                r"\b(work(s|ed|ing)?) together\b"
            ],
            
            "regulatory": [
                r"\b(regulat(ory|ion|or)|complian(t|ce)|authority|commission|approval)\b",
                r"\b(approve[ds]?|authoriz(e[ds]?)|grant(s|ed)?) .{1,20}\b(by|from) .{1,15}\b(regulat(or|ory)|authority|commission)\b",
                r"\b(reject(s|ed)?|den(y|ies|ied)|block(s|ed)?) .{1,20}\b(by|from) .{1,15}\b(regulat(or|ory)|authority|commission)\b",
                r"\b(regulatory|compliance) (requirement|standard|framework)\b",
                r"\b(SEC|FDA|EU Commission|FTC|authority|FSA|regulator)\b"
            ],
            
            "stock_movement": [
                r"\b(stock|share|equity) (price|value|trading)\b",
                r"\b(trading|trade[ds]?|exchange|market) .{1,20}\b(stock|share|securities)\b",
                r"\b(rise[ds]?|drop(s|ped)?|fall[s]?|fell|gain(s|ed)?|increas(e[ds]?|ing)|decreas(e[ds]?|ing)|jump(s|ed)?|plunge[ds]?|plummet(s|ed)?)\b .{1,20}\b(stock|share|price|value|market|index|points?)\b",
                r"\b(bull|bear|volatile|stable) market\b",
                r"\b(buy|sell) signal\b",
                r"\b(index|NASDAQ|DOW|S&P|NYSE)\b",
                r"\btarget .{1,15}\b(price|level)\b"
            ],
            
            "debt_financing": [
                r"\b(debt|loan|bond|credit|financing)\b",
                r"\b(rais(e[ds]?|ing)|secur(e[ds]?|ing)) .{1,20}\b(capital|debt|loan|financing|funds)\b",
                r"\b(issue[ds]?|sell[s]?|sold) .{1,20}\b(bond|note|debt|securities)\b",
                r"\b(credit|loan|debt) (facility|agreement|covenant|terms)\b",
                r"\b(borrow(s|ed|ing)?|lend(s|ed|ing)?|financ(e[ds]?|ing)?)\b"
            ],
            
            "contract_deal": [
                r"\b(contract|deal|agreement|arrangement|negotiations?)\b",
                r"\b(sign(s|ed)?|win(s)?|won|secur(e[ds]?)|award(s|ed)?) .{1,20}\b(contract|deal|agreement)\b",
                r"\b(multi[- ]year|long[- ]term) (agreement|contract|deal)\b",
                r"\b(worth|valued at) .{1,20}\b(million|billion|mn|bn)\b",
                r"\b(supply|deliver|provide)(s|ed|ing)? .{1,20}\b(to|for)\b",
                r"\bfinaliz(e|ing) .{1,20}\bnegotiations\b",
                r"\bsign .{1,20}\bcontracts?\b"
            ]
        }
    
    FINANCIAL_CONTEXT_CLUES = {
            "merger_acquisition": {
                'strong': ['acquiring', 'merged', 'acquisition', 'takeover', 'buyout', 'stake', 'ownership', 'combining'],
                'moderate': ['transaction', 'deal', 'synergy', 'integration', 'buyer', 'seller'],
                'weak': ['strategic', 'growth', 'combined', 'entity']
            },
            "earnings": {
                'strong': ['revenue', 'profit', 'loss', 'earnings', 'financial', 'results', 'quarterly', 'reported'],
                'moderate': ['fiscal', 'year', 'quarter', 'guidance', 'forecast', 'performance'],
                'weak': ['compared', 'previous', 'period', 'increase', 'decrease']
            },
            "dividend": {
                'strong': ['dividend', 'payout', 'shareholder', 'distribution', 'yield'],
                'moderate': ['quarterly', 'annual', 'payment', 'per share', 'declaration'],
                'weak': ['return', 'investor', 'income']
            },
            "product_launch": {
                'strong': ['launch', 'new product', 'introducing', 'release', 'unveiled', 'innovation'],
                'moderate': ['features', 'customers', 'market', 'solution', 'technology'],
                'weak': ['design', 'quality', 'improved', 'performance']
            },
            "investment": {
                'strong': ['invest', 'funding', 'capital', 'million', 'billion', 'raised'],
                'moderate': ['venture', 'equity', 'financing', 'round', 'investor'],
                'weak': ['growth', 'development', 'expansion', 'strategy']
            },
            "restructuring": {
                'strong': ['restructuring', 'reorganization', 'cost-cutting', 'downsizing', 'streamline'],
                'moderate': ['efficiency', 'savings', 'transform', 'optimization', 'consolidation'],
                'weak': ['plan', 'initiative', 'program', 'changes', 'improvement']
            },
            "litigation": {
                'strong': ['lawsuit', 'legal', 'court', 'litigation', 'sue', 'settlement', 'dispute'],
                'moderate': ['claim', 'damages', 'plaintiff', 'defendant', 'judge', 'allegations'],
                'weak': ['case', 'issue', 'matter', 'proceeding', 'resolution']
            },
            "executive_change": {
                'strong': ['CEO', 'CFO', 'CTO', 'COO', 'appointed', 'resigned', 'leadership', 'executive'],
                'moderate': ['management', 'board', 'director', 'chairman', 'president', 'successor'],
                'weak': ['lead', 'role', 'position', 'responsibility', 'team']
            },
            "expansion": {
                'strong': ['expansion', 'growing', 'new market', 'facility', 'opening', 'international'],
                'moderate': ['presence', 'global', 'region', 'country', 'location', 'footprint'],
                'weak': ['growth', 'opportunity', 'strategic', 'position', 'strengthen']
            },
            "layoff": {
                'strong': ['layoff', 'job cut', 'redundancy', 'workforce reduction', 'downsizing'],
                'moderate': ['employees', 'workers', 'staff', 'positions', 'terminate', 'eliminate'],
                'weak': ['cost', 'efficiency', 'restructuring', 'consolidation']
            },
            "partnership": {
                'strong': ['partnership', 'alliance', 'collaboration', 'joint venture', 'cooperate'],
                'moderate': ['partner', 'strategic', 'together', 'combined', 'mutual'],
                'weak': ['agreement', 'opportunity', 'synergy', 'relationship']
            },
            "regulatory": {
                'strong': ['regulatory', 'approval', 'regulator', 'compliance', 'regulation'],
                'moderate': ['authority', 'commission', 'agency', 'approved', 'permitted'],
                'weak': ['requirement', 'standard', 'guideline', 'framework', 'rule']
            },
            "stock_movement": {
                'strong': ['stock', 'share', 'price', 'trading', 'market', 'rose', 'fell'],
                'moderate': ['investor', 'exchange', 'value', 'index', 'points', 'percent'],
                'weak': ['volatility', 'trend', 'performance', 'sentiment']
            },
            "debt_financing": {
                'strong': ['debt', 'loan', 'bond', 'financing', 'credit', 'borrowing'],
                'moderate': ['facility', 'term', 'interest', 'maturity', 'lender', 'refinance'],
                'weak': ['capital', 'structure', 'balance', 'sheet', 'liquidity']
            },
            "contract_deal": {
                'strong': ['contract', 'deal', 'agreement', 'signed', 'awarded', 'client'],
                'moderate': ['worth', 'value', 'million', 'billion', 'terms', 'project'],
                'weak': ['service', 'supply', 'deliver', 'period', 'customer']
            }
        }
    



    INDUSTRY_SECTORS = [
        "technology",          # Software, hardware, IT services, semiconductors, etc.
        "healthcare",          # Pharmaceuticals, medical devices, hospitals, biotech, etc.
        "finance",             # Banking, insurance, investment, fintech, etc.
        "energy",              # Oil, gas, renewables, utilities, power generation, etc.
        "consumer_goods",      # Retail, apparel, food products, household goods, etc.
        "manufacturing",       # Industrial equipment, machinery, factories, etc.
        "telecommunications",  # Wireless carriers, internet service, network equipment, etc.
        "transportation",      # Airlines, shipping, railways, automotive, logistics, etc.
        "real_estate",         # Property development, REITs, construction, etc.
        "media_entertainment", # Publishing, broadcasting, streaming, gaming, etc.
        "materials",           # Chemicals, mining, metals, forestry, etc.
        "agriculture"          # Farming, food production, agricultural products, etc.
    ]
    
    # Keywords associated with each industry sector
    INDUSTRY_KEYWORDS = {
        "technology": [
            "software", "hardware", "tech", "technology", "semiconductor", "chip", "computer", 
            "internet", "cloud", "digital", "IT", "information technology", "app", "application",
            "mobile", "electronic", "computing", "artificial intelligence", "AI", "machine learning", 
            "data", "cybersecurity", "security", "e-commerce", "platform", "algorithm", 
            "SaaS", "PaaS", "IaaS", "blockchain", "quantum", "robotics", "automation", 
            "IoT", "internet of things", "network", "website", "online", "virtual", "web", 
            "developer", "coding", "programming", "startup", "tech company", "processor", 
            "memory", "storage", "database", "software as a service", "big data", "analytics"
        ],
        
        "healthcare": [
            "healthcare", "health", "medical", "medicine", "pharmaceutical", "pharma", "biotech", 
            "biotechnology", "drug", "therapy", "therapeutic", "vaccine", "clinical", "hospital", 
            "doctor", "physician", "patient", "treatment", "disease", "diagnostic", "health insurance", 
            "medtech", "medical device", "health system", "care", "clinic", "surgery", "telehealth", 
            "life science", "genomic", "DNA", "RNA", "protein", "antibody", "molecule", "oncology", 
            "cancer", "cardio", "heart", "orthopedic", "neurology", "brain", "elderly care", 
            "mental health", "wellness", "fitness", "medical equipment", "laboratory", "FDA", 
            "EMA", "clinical trial", "phase", "approval", "prescription", "generic drug"
        ],
        
        "finance": [
            "bank", "banking", "finance", "financial", "investment", "investor", "invest", 
            "loan", "lending", "credit", "mortgage", "insurance", "insurer", "wealth", 
            "asset management", "hedge fund", "private equity", "venture capital", "VC", 
            "capital market", "stock market", "exchange", "trading", "trader", "brokerage", 
            "broker", "fintech", "payment", "transaction", "money", "currency", "cash", 
            "deposit", "savings", "checking", "account", "interest rate", "dividend", 
            "treasury", "bond", "debt", "equity", "share", "stock", "securities", 
            "derivative", "option", "future", "commodity", "crypto", "cryptocurrency", 
            "bitcoin", "blockchain finance", "pension", "retirement", "fund", "portfolio"
        ],
        
        "energy": [
            "energy", "power", "electricity", "gas", "oil", "petroleum", "crude", "refinery", 
            "fuel", "drilling", "well", "reservoir", "fossil fuel", "coal", "natural gas", 
            "renewable energy", "solar", "wind", "hydro", "hydroelectric", "geothermal", 
            "biomass", "biofuel", "nuclear", "utility", "utilities", "power plant", "generator", 
            "turbine", "pipeline", "transmission", "distribution", "grid", "battery", "storage", 
            "carbon", "emissions", "climate", "clean energy", "green energy", "sustainable energy", 
            "energy efficiency", "electric", "power generation", "exploration", "production", 
            "upstream", "midstream", "downstream", "oilfield", "rig", "shale", "LNG", 
            "liquefied natural gas", "OPEC", "barrel", "watt", "megawatt", "gigawatt"
        ],
        
        "consumer_goods": [
            "retail", "retailer", "consumer", "customer", "product", "brand", "goods", 
            "merchandise", "shopping", "store", "mall", "shop", "supermarket", "grocery", 
            "food", "beverage", "drink", "apparel", "clothing", "fashion", "textile", 
            "footwear", "shoes", "luxury", "cosmetic", "beauty", "personal care", "household", 
            "furniture", "appliance", "electronics", "smartphone", "device", "gadget", 
            "e-commerce", "online shopping", "delivery", "supply chain", "inventory", 
            "distribution", "wholesale", "discount", "price", "consumer staples", 
            "consumer discretionary", "FMCG", "fast moving consumer goods", "CPG", 
            "consumer packaged goods", "department store", "specialty store", "franchise", 
            "chain", "direct-to-consumer", "D2C", "omnichannel"
        ],
        
        "manufacturing": [
            "manufacturing", "manufacturer", "factory", "plant", "production", "industrial", 
            "industry", "machinery", "machine", "equipment", "tool", "component", "part", 
            "assembly", "fabrication", "processing", "supply chain", "inventory", "raw material", 
            "engineering", "design", "prototype", "quality control", "inspection", "automation", 
            "robotics", "production line", "conveyor", "forge", "casting", "molding", "welding", 
            "cutting", "drilling", "industrial equipment", "heavy machinery", "light industry", 
            "heavy industry", "aerospace", "defense", "military", "contract manufacturing", 
            "OEM", "original equipment manufacturer", "CNC", "computer numerical control", 
            "3D printing", "additive manufacturing", "injection molding", "ISO", "lean manufacturing", 
            "just-in-time", "JIT", "six sigma", "kaizen", "labor", "workforce", "union"
        ],
        
        "telecommunications": [
            "telecom", "telecommunication", "communications", "phone", "mobile", "cellular", 
            "wireless", "broadband", "internet", "network", "connectivity", "carrier", 
            "provider", "service provider", "ISP", "internet service provider", "fiber", 
            "optical", "cable", "satellite", "tower", "infrastructure", "spectrum", "frequency", 
            "bandwidth", "data", "voice", "text", "call", "messaging", "roaming", "5G", "4G", 
            "LTE", "3G", "GSM", "CDMA", "VoIP", "landline", "fixed-line", "telco", "router", 
            "modem", "switch", "networking equipment", "base station", "cell tower", "antenna", 
            "transmission", "reception", "signal", "coverage", "mobile network operator", "MNO", 
            "virtual network", "MVNO", "unified communications", "telecommunications act"
        ],
        
        "transportation": [
            "transport", "transportation", "logistics", "shipping", "freight", "cargo", 
            "delivery", "courier", "airline", "aviation", "aircraft", "airplane", "flight", 
            "airport", "railway", "railroad", "train", "locomotive", "rail", "track", 
            "automotive", "auto", "automobile", "car", "vehicle", "truck", "trucking", 
            "fleet", "marine", "maritime", "ship", "vessel", "container", "port", "harbor", 
            "terminal", "warehouse", "distribution center", "fulfillment", "transit", 
            "passenger", "travel", "journey", "ride", "ride-sharing", "last mile", 
            "intermodal", "multimodal", "supply chain", "route", "network", "infrastructure", 
            "highway", "road", "bridge", "tunnel", "electric vehicle", "EV", "autonomous", 
            "self-driving", "drone", "metro", "subway", "bus", "commercial vehicle"
        ],
        
        "real_estate": [
            "real estate", "property", "building", "construction", "development", "developer", 
            "commercial property", "residential property", "housing", "house", "home", 
            "apartment", "condo", "condominium", "office", "retail space", "industrial space", 
            "land", "lot", "real property", "architecture", "design", "renovation", 
            "remodeling", "lease", "rent", "rental", "tenant", "landlord", "property management", 
            "facility management", "REIT", "real estate investment trust", "mortgage", 
            "financing", "appraisal", "valuation", "zoning", "urban planning", "mixed-use", 
            "square foot", "square meter", "amenity", "real estate agent", "broker", 
            "listing", "closing", "title", "deed", "escrow", "foreclosure", "commercial real estate", 
            "CRE", "residential real estate", "homebuilder", "subdivision", "community", 
            "property tax", "real estate market", "housing market"
        ],
        
        "media_entertainment": [
            "media", "entertainment", "content", "publisher", "publishing", "broadcast", 
            "broadcasting", "network", "television", "TV", "radio", "film", "movie", 
            "cinema", "studio", "production", "post-production", "streaming", "digital media", 
            "social media", "platform", "channel", "video", "audio", "music", "record", 
            "recording", "label", "artist", "performer", "celebrity", "talent", "concert", 
            "live event", "theater", "stage", "venue", "ticket", "box office", "audience", 
            "viewer", "listener", "subscriber", "subscription", "advertising", "ad", 
            "commercial", "marketing", "promotion", "sponsorship", "news", "journalism", 
            "press", "newspaper", "magazine", "digital publication", "blog", "gaming", 
            "game", "video game", "esports", "sports", "league", "team", "franchise"
        ],
        
        "materials": [
            "materials", "chemical", "chemistry", "compound", "substance", "element", 
            "mineral", "mining", "miner", "extraction", "metal", "metallurgy", "steel", 
            "iron", "aluminum", "copper", "gold", "silver", "precious metal", "alloy", 
            "polymer", "plastic", "resin", "fiber", "composite", "ceramic", "glass", 
            "concrete", "cement", "aggregate", "stone", "rock", "granite", "marble", 
            "lumber", "timber", "wood", "forestry", "forest product", "paper", "pulp", 
            "packaging", "container", "industrial gas", "fertilizer", "pesticide", 
            "commodity", "raw material", "refining", "processing", "specialty chemical", 
            "petrochemical", "petroleum product", "coal", "mineral resource", "ore", 
            "mine", "quarry", "pit", "processing plant", "smelter", "refinery"
        ],
        
        "agriculture": [
            "agriculture", "agricultural", "farming", "farm", "farmer", "crop", "harvest", 
            "cultivation", "livestock", "cattle", "dairy", "poultry", "meat", "fishery", 
            "aquaculture", "seafood", "grain", "corn", "wheat", "soybean", "rice", 
            "fruit", "vegetable", "organic", "sustainable agriculture", "agribusiness", 
            "food processing", "food production", "seed", "feed", "fertilizer", "pesticide", 
            "irrigation", "equipment", "tractor", "machinery", "soil", "land", "plantation", 
            "greenhouse", "vertical farming", "precision agriculture", "agtech", "agricultural technology", 
            "rural", "commodity", "futures", "yield", "productivity", "GMO", "genetically modified", 
            "biotechnology", "breeding", "hybrid", "pest control", "weed control", "herbicide", 
            "fungicide", "insecticide", "cooperative", "subsidies", "agriculture policy"
        ]
    }
    
    # Context patterns for better industry detection
    INDUSTRY_CONTEXT_PATTERNS = {
        "technology": [
            (r"(technology|tech|software|digital|IT).{1,30}(company|firm|sector|industry|business|giant)", 2.0),
            (r"(develop\w*|launch\w*|release\w*).{1,30}(software|app|platform|solution|product|service).{1,30}(for|to|that)", 2.0),
            (r"(AI|artificial intelligence|machine learning|ML|deep learning|algorithm).{1,30}(to|for|that|which).{1,30}(predict|analyze|process|improve)", 2.5),
            (r"(cloud|SaaS|software as a service|platform as a service|PaaS).{1,30}(solution|service|product|offering|provider)", 2.0),
            (r"(data|big data|analytics).{1,30}(company|firm|provider|solution|platform|technology)", 1.5),
            (r"(silicon|chip|semiconductor|processor).{1,30}(design|manufacturing|production|technology|company|shortage|supply)", 2.5)
        ],
        
        "healthcare": [
            (r"(health|healthcare|medical|pharmaceutical|biotech).{1,30}(company|firm|sector|industry|provider|system)", 2.0),
            (r"(drug|medicine|therapy|treatment|vaccine|device).{1,30}(for|to treat|to prevent|used in).{1,30}(disease|condition|disorder|cancer|diabetes|patients)", 2.5),
            (r"(clinical|trial|study|research).{1,30}(for|of|on|to).{1,30}(drug|therapy|treatment|medicine|vaccine|device)", 2.0),
            (r"(FDA|EMA|regulatory).{1,30}(approval|clearance|authorization|submission|review|decision).{1,30}(for|of|on).{1,30}(drug|device|therapy|treatment)", 2.5),
            (r"(hospital|clinic|physician|doctor|healthcare provider).{1,30}(care|service|patient|treatment|facility)", 1.5),
            (r"(biotechnology|life science|genomic|genetic).{1,30}(research|technology|company|firm|development|advance)", 2.0)
        ],
        
        "finance": [
            (r"(bank|financial|investment|insurance).{1,30}(company|firm|institution|sector|industry|group)", 2.0),
            (r"(loan|mortgage|credit|deposit|debt|lending).{1,30}(market|business|portfolio|growth|decline|rate)", 2.0),
            (r"(stock|share|equity|bond|security|market).{1,30}(exchange|trading|performance|index|price|value)", 2.5),
            (r"(investor|investment|fund|asset|portfolio|capital).{1,30}(manage|allocate|return|value|performance|strategy)", 2.0),
            (r"(fintech|financial technology|payment|transaction).{1,30}(system|platform|solution|company|service|provider)", 2.0),
            (r"(banking|investment banking|wealth management|insurance).{1,30}(service|business|division|operation|sector)", 1.5)
        ],
        
        "energy": [
            (r"(energy|power|oil|gas|electricity|fuel).{1,30}(company|firm|sector|industry|producer|provider|supplier)", 2.0),
            (r"(oil|gas|petroleum|crude).{1,30}(production|exploration|drilling|field|well|reserve|price)", 2.5),
            (r"(renewable|solar|wind|hydro|clean).{1,30}(energy|power|electricity|project|capacity|generation|source)", 2.5),
            (r"(power|electric|electricity).{1,30}(plant|station|generation|grid|utility|supply|distribution|transmission)", 2.0),
            (r"(energy|power).{1,30}(consumption|supply|demand|price|cost|efficiency|transition|policy)", 1.5),
            (r"(carbon|emission|climate|green|sustainable).{1,30}(energy|fuel|power|technology|solution|goal|target)", 2.0)
        ],
        
        "consumer_goods": [
            (r"(retail|consumer|product|brand|goods).{1,30}(company|firm|sector|industry|business|market)", 2.0),
            (r"(store|shop|outlet|mall|retail|merchandise).{1,30}(chain|location|sale|customer|shopping|experience)", 2.0),
            (r"(food|beverage|drink|grocery|supermarket).{1,30}(product|brand|store|chain|market|company|manufacturer)", 2.5),
            (r"(clothing|apparel|fashion|footwear|accessory|textile).{1,30}(brand|retailer|company|manufacturer|designer|product)", 2.0),
            (r"(consumer|household|personal care|beauty|cosmetic).{1,30}(product|goods|brand|item|market|sector)", 2.0),
            (r"(e-commerce|online|digital).{1,30}(retail|shopping|store|marketplace|platform|channel|seller)", 2.0)
        ],
        
        "manufacturing": [
            (r"(manufacturing|industrial|factory|production).{1,30}(company|firm|sector|industry|plant|facility)", 2.0),
            (r"(equipment|machinery|component|part|product).{1,30}(production|manufacturing|assembly|fabrication|design)", 2.0),
            (r"(industrial|manufacturing).{1,30}(automation|process|system|technology|solution|efficiency|output)", 2.5),
            (r"(supply chain|production|manufacturing).{1,30}(issue|problem|challenge|disruption|shortage|constraint)", 2.0),
            (r"(manufacturer|producer|maker).{1,30}(of|for).{1,30}(equipment|machinery|part|component|product|material)", 2.5),
            (r"(manufacturing|production|assembly|fabrication).{1,30}(capacity|capability|facility|site|plant|line)", 1.5)
        ],
        
        "telecommunications": [
            (r"(telecom|telecommunications|communications|wireless|mobile).{1,30}(company|firm|provider|carrier|operator|sector)", 2.0),
            (r"(network|infrastructure|service|connectivity).{1,30}(provider|operator|carrier|company|supplier)", 2.0),
            (r"(5G|4G|broadband|fiber|wireless).{1,30}(network|technology|infrastructure|service|deployment|coverage|rollout)", 2.5),
            (r"(mobile|cell|phone|telephone|data|voice).{1,30}(service|plan|subscription|customer|usage|traffic)", 2.0),
            (r"(internet|broadband|connectivity|connection).{1,30}(service|access|speed|provider|quality|reliability)", 2.0),
            (r"(spectrum|frequency|bandwidth|tower|satellite).{1,30}(allocation|auction|license|capacity|coverage|infrastructure)", 2.0)
        ],
        
        "transportation": [
            (r"(transport|transportation|logistics|shipping|freight).{1,30}(company|firm|provider|carrier|operator|sector)", 2.0),
            (r"(airline|air|aviation|flight|aircraft).{1,30}(company|carrier|operator|service|industry|travel)", 2.5),
            (r"(shipping|freight|cargo|container|maritime).{1,30}(company|service|vessel|ship|carrier|route|port)", 2.0),
            (r"(rail|railroad|railway|train).{1,30}(company|operator|service|network|line|station|transportation)", 2.0),
            (r"(car|auto|vehicle|automotive).{1,30}(manufacturer|maker|company|industry|production|market|sales)", 2.5),
            (r"(logistics|delivery|supply chain|fulfillment).{1,30}(service|company|provider|network|solution|operation)", 2.0)
        ],
        
        "real_estate": [
            (r"(real estate|property|building|housing).{1,30}(company|firm|developer|investor|market|sector)", 2.0),
            (r"(commercial|residential|office|retail).{1,30}(property|real estate|space|building|development|project)", 2.5),
            (r"(construction|development|building).{1,30}(project|company|firm|activity|sector|industry|work)", 2.0),
            (r"(lease|rent|rental|tenant|landlord).{1,30}(property|space|building|agreement|market|rate|income)", 2.0),
            (r"(REIT|real estate investment trust|property fund).{1,30}(portfolio|investment|asset|acquisition|share|dividend)", 2.5),
            (r"(housing|home|mortgage|property).{1,30}(market|price|value|sales|demand|supply|loan|rate)", 2.0)
        ],
        
        "media_entertainment": [
            (r"(media|entertainment|content|publishing|broadcast).{1,30}(company|firm|business|industry|sector|group)", 2.0),
            (r"(TV|television|video|streaming|film|movie).{1,30}(content|service|platform|studio|production|show|series)", 2.5),
            (r"(music|audio|game|gaming|sports).{1,30}(company|service|platform|content|production|publisher|studio)", 2.0),
            (r"(news|publishing|magazine|newspaper).{1,30}(outlet|organization|company|publisher|content|platform)", 2.0),
            (r"(advertising|marketing|ad).{1,30}(revenue|business|market|campaign|agency|platform|spend)", 2.0),
            (r"(digital|online|social).{1,30}(media|content|platform|service|channel|audience|creator)", 2.0)
        ],
        
        "materials": [
            (r"(materials|chemical|metal|mining).{1,30}(company|firm|producer|supplier|manufacturer|sector)", 2.0),
            (r"(mining|extraction|production).{1,30}(of|for).{1,30}(metal|mineral|ore|resource|material|commodity)", 2.5),
            (r"(steel|iron|aluminum|copper|gold|silver).{1,30}(producer|production|manufacturing|processing|company|market|price)", 2.0),
            (r"(chemical|specialty chemical|petrochemical).{1,30}(company|producer|manufacturer|production|plant|product)", 2.0),
            (r"(forest|timber|wood|paper|pulp).{1,30}(product|company|producer|industry|resource|processing)", 2.0),
            (r"(material|raw material|commodity).{1,30}(price|cost|supply|demand|shortage|constraint|market)", 2.0)
        ],
        
        "agriculture": [
            (r"(agriculture|agricultural|farming|farm).{1,30}(company|business|sector|industry|producer|operation)", 2.0),
            (r"(crop|grain|produce|food).{1,30}(production|producer|growing|harvest|yield|farmer|processing)", 2.5),
            (r"(livestock|cattle|dairy|poultry|meat).{1,30}(production|producer|farm|farmer|company|industry)", 2.0),
            (r"(seed|fertilizer|pesticide|agricultural).{1,30}(company|producer|supplier|product|technology|market)", 2.0),
            (r"(food|grain|crop).{1,30}(price|market|production|supply|demand|export|import|trade)", 2.0),
            (r"(sustainable|organic|precision).{1,30}(agriculture|farming|food|crop|production|practice|method)", 2.0)
        ]
    }
    
    # Combinations of industries that commonly overlap
    INDUSTRY_OVERLAPS = {
        ("technology", "healthcare"): "health tech, medical technology, healthcare IT",
        ("technology", "finance"): "fintech, financial technology, digital banking",
        ("technology", "media_entertainment"): "digital media, streaming platforms, content technology",
        ("healthcare", "manufacturing"): "medical device manufacturing, pharmaceutical production",
        ("energy", "materials"): "petrochemicals, mining for energy resources",
        ("consumer_goods", "agriculture"): "food production, agricultural products for consumers",
        ("manufacturing", "transportation"): "automotive manufacturing, aerospace production",
        ("real_estate", "finance"): "real estate investment, property financing"
    }
    
    # Entity patterns for company and industry identification
    INDUSTRY_ENTITY_PATTERNS = {
        "company_name": r"([A-Z][a-z]*\.?\s)?([A-Z][a-z]+\s)*[A-Z][a-z]*\.?(\s(Inc|Corp|Co|Ltd|LLC|Group|SA|AG|SE|NV|PLC|GmbH)\.?)?",
        "ticker_symbol": r"\$[A-Z]{1,5}",
        "industry_terms": r"(sector|industry|segment|market|business)"
    }

    
    # Financial performance indicators
    PERFORMANCE_INDICATORS = {
        'earnings_beat': ['beat', 'exceeded', 'above', 'surpassed', 'better than expected'],
        'earnings_miss': ['miss', 'missed', 'below', 'disappointing', 'worse than expected'],
        'guidance_up': ['raise', 'raised', 'boost', 'boosted', 'increase', 'increased', 'higher', 'optimistic'],
        'guidance_down': ['lower', 'cut', 'reduce', 'reduced', 'decrease', 'decreased', 'pessimistic', 'cautious']
    }
    
    # Regex patterns for financial entities
    COMPANY_PATTERN = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'
    TICKER_PATTERN = r'\$[A-Z]{1,5}|\b[A-Z]{2,5}\b'
    PERCENTAGE_PATTERN = r'(?:(?:\+|\-)?(?:\d+(?:\.\d+)?|\.\d+)\s*%)'
    CURRENCY_AMOUNT_PATTERN = r'(?:(?:\$|€|£|¥)?\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|m|b|t))?(?:\s*(?:\$|€|£|¥))?)'