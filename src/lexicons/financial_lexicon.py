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
    
    # Define the 15 most common financial events
    FINANCIAL_EVENTS = [
        "merger_acquisition",   # Mergers, acquisitions, takeovers
        "earnings",             # Earnings reports, profit/loss statements
        "dividend",             # Dividend announcements, changes
        "product_launch",       # New product/service announcements
        "investment",           # Investments, funding rounds
        "restructuring",        # Restructuring, reorganization
        "litigation",           # Legal proceedings, lawsuits
        "executive_change",     # Leadership changes, executive appointments
        "expansion",            # Expansion to new markets, opening facilities
        "layoff",               # Layoffs, job cuts
        "partnership",          # Strategic partnerships, collaborations
        "regulatory",           # Regulatory issues, approvals, compliance
        "stock_movement",       # Stock price changes, trading activity
        "debt_financing",       # Debt issues, loans, bonds
        "contract_deal"         # New contracts or deals
    ]


    FINANCIAL_EVENT_PATTERNS = {
            "merger_acquisition": [
                r"(merger|acquisition|takeover|acquire|merge|bid|buyout|consolidat)",
                r"(purchas\w+|bought|buy) (company|stake|share|business)",
                r"(take\w+) (over|control)",
                r"(percent|%) stake",
                r"(join\w+) forces"
            ],
            
            "earnings": [
                r"(earnings|revenue|profit|loss|income|EBITDA|EPS)",
                r"(financial|quarterly|annual) results",
                r"(report\w+) (profit|loss|revenue)",
                r"(operat\w+) (profit|loss|margin)",
                r"(net|gross) (income|profit|loss)",
                r"dividend per share",
                r"fiscal (year|quarter)"
            ],
            
            "dividend": [
                r"dividend\w*",
                r"(declar\w+|announce\w+|raise\w+|increase\w+|cut\w+|reduc\w+|suspend\w+) dividend",
                r"(pay\w+|distribut\w+) (to (share|stock)holders)",
                r"(quarterly|annual|special) (payment|distribution)",
                r"payout ratio",
                r"yield"
            ],
            
            "product_launch": [
                r"(launch|introduc\w+|unveil\w+|releas\w+) (new|product|service)",
                r"new (product|service|solution|platform|technology)",
                r"(product|range) (portfolio|line|offering)",
                r"(next(\s|-|)generation|cutting(\s|-|)edge)",
                r"(begin|start\w+) (production|manufacturing)"
            ],
            
            "investment": [
                r"(invest\w+|funding|capital|raised)",
                r"(venture|private equity|seed) (capital|investment|funding)",
                r"(series|round) [A-Z]",
                r"(million|billion|mn|bn|EUR|USD|€|\$)",
                r"(shareholding|holding)",
                r"(fund\w+) (by|from)"
            ],
            
            "restructuring": [
                r"(restructur\w+|reorganiz\w+|transform\w+)",
                r"(cost|operational) (reduction|cutting|saving)",
                r"(efficiency|streamlining) (program|measure|initiative)",
                r"(business|operational) (model|structure)",
                r"(divest\w+|sell\w+) (unit|division|business)"
            ],
            
            "litigation": [
                r"(litigation|lawsuit|legal|court|dispute|sue\w+)",
                r"(legal|court) (proceedings|case|battle|fight)",
                r"(antitrust|regulatory) (investigation|probe)",
                r"(settle\w+|resolution) (case|dispute|claim)",
                r"(fine\w+|penalty|damages)"
            ],
            
            "executive_change": [
                r"(CEO|CFO|CTO|COO|chief|executive|officer|director|board)",
                r"(appoint\w+|hire\w+|name\w+|elect\w+) (as|to) (CEO|CFO|CTO|COO|chief|executive|position)",
                r"(management|leadership) (change|team)",
                r"(resign\w+|depart\w+|step\w+ down|leaving)",
                r"(succeed\w+|replace\w+) (as|by)"
            ],
            
            "expansion": [
                r"(expan\w+|grow\w+) (into|market|business|operations)",
                r"(new|international|global) (market|facility|location|office|store)",
                r"(open\w+|establish\w+|launch\w+) (office|facility|store|presence)",
                r"(entry|expansion) (into|in) ([A-Z]\w+)",
                r"(global|international|worldwide) (reach|presence|footprint)"
            ],
            
            "layoff": [
                r"(layoff\w*|downsize\w*|job cut\w*|cut\w* job|redundanc\w*)",
                r"(reduc\w+|cut\w+) (workforce|staff|employees|personnel|headcount)",
                r"(eliminate|shed) (position|job)",
                r"(employee|worker|staff) (reduction|termination)",
                r"(lose|lost|loose) their jobs"
            ],
            
            "partnership": [
                r"(partnership|collaboration|alliance|joint venture)",
                r"(partner\w+|collaborat\w+|team\w+ up) with",
                r"(strategic|key) (relationship|partnership|alliance)",
                r"(sign\w+|enter\w+|form\w+) (agreement|deal|partnership)",
                r"(work\w+) together"
            ],
            
            "regulatory": [
                r"(regulat\w+|compli\w+|authority|commission|approval)",
                r"(approve\w+|authorize\w+|grant\w+) (by|from) (regulator|authority|commission)",
                r"(reject\w+|deny\w+|block\w+) (by|from) (regulator|authority|commission)",
                r"(regulatory|compliance) (requirement|standard|framework)",
                r"(SEC|FDA|EU|Commission|FTC|authority)"
            ],
            
            "stock_movement": [
                r"(stock|share|equity) (price|value|market)",
                r"(trading|trade|exchange|market)",
                r"(rise|drop|fall|gain|increase|decrease|jump|plunge|plummet)",
                r"(bull|bear|volatile|stable) market",
                r"(buy|sell) signal",
                r"(index|nasdaq|dow|S&P)"
            ],
            
            "debt_financing": [
                r"(debt|loan|bond|credit|financing)",
                r"(raise\w+|secure\w+) (capital|debt|loan|financing)",
                r"(issue\w+|sell\w+) (bond|note|debt)",
                r"(credit|loan|debt) (facility|agreement|covenant)",
                r"(borrow\w+|lend\w+|finance\w+)"
            ],
            
            "contract_deal": [
                r"(contract|deal|agreement|arrangement)",
                r"(sign\w+|win\w+|secure\w+|award\w+) (contract|deal)",
                r"(multi-year|long-term) (agreement|contract|deal)",
                r"(worth|valued at) (million|billion|mn|bn)",
                r"(supply|deliver|provide) (to|for)"
            ]
        }

    # # Financial events
    # FINANCIAL_EVENTS = {
    #     'earnings': ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'report', 'reported'],
    #     'merger_acquisition': ['merger', 'acquisition', 'acquire', 'acquired', 'buyout', 'takeover'],
    #     'product_launch': ['launch', 'launched', 'introduce', 'introduced', 'unveil', 'unveiled', 'release', 'released'],
    #     'leadership_change': ['ceo', 'executive', 'appoint', 'appointed', 'resign', 'resigned', 'management'],
    #     'regulatory': ['regulation', 'regulatory', 'compliance', 'regulator', 'sec', 'fined', 'penalty'],
    #     'investment': ['invest', 'investment', 'funding', 'fund', 'capital', 'venture', 'stake'],
    #     'restructuring': ['restructure', 'restructuring', 'layoff', 'layoffs', 'downsize', 'downsizing', 'cost-cutting'],
    #     'dividend': ['dividend', 'dividends', 'payout', 'yield', 'shareholder', 'stockholder'],
    #     'litigation': ['lawsuit', 'litigation', 'legal', 'sue', 'sued', 'court', 'settlement'],
    #     'market_expansion': ['expansion', 'expand', 'global', 'international', 'enter', 'entered', 'market']
    # }
    
    # Industries/Sectors
    INDUSTRIES = {
        'technology': ['tech', 'software', 'hardware', 'ai', 'artificial intelligence', 'cloud', 'computing', 'semiconductor', 'internet'],
        'finance': ['bank', 'financial', 'insurance', 'fintech', 'credit', 'loan', 'mortgage', 'invest', 'asset'],
        'healthcare': ['health', 'medical', 'pharma', 'biotech', 'drug', 'therapeutics', 'hospital', 'patient'],
        'energy': ['oil', 'gas', 'energy', 'renewable', 'solar', 'wind', 'petroleum', 'power', 'utility'],
        'telecom': ['telecom', 'communication', 'wireless', 'broadband', 'network', 'mobile', 'phone'],
        'automotive': ['auto', 'automotive', 'vehicle', 'car', 'electric', 'ev', 'transportation', 'mobility'],
        'materials': ['materials', 'mining', 'metal', 'steel', 'chemicals', 'paper', 'packaging', 'commodity'],
        'media': ['media', 'entertainment', 'streaming', 'content', 'advertising', 'publishing', 'broadcast'],
        'realestate': ['real estate', 'property', 'reit', 'housing', 'commercial', 'residential', 'development'],
        'retail': ['retail', 'store', 'shop', 'consumer', 'mall', 'outlet', 'e-commerce', 'merchandise', 'brand', 'product', 'customer', 'sales'],
        'manufacturing': ['manufacturing', 'factory', 'production', 'industrial', 'machinery', 'equipment', 'assembly', 'fabrication', 'engineering', 'raw material', 'aerospace', 'defense', 'chemical'],
        'services': ['service', 'consulting', 'outsourcing', 'professional', 'support', 'solution', 'assistance', 'client', 'provider'],
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