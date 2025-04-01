class FinancialLexicon:
    """Financial lexicons and gazetteers for feature engineering"""
    
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
    
    # Financial events
    FINANCIAL_EVENTS = {
        'earnings': ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'report', 'reported'],
        'merger_acquisition': ['merger', 'acquisition', 'acquire', 'acquired', 'buyout', 'takeover'],
        'product_launch': ['launch', 'launched', 'introduce', 'introduced', 'unveil', 'unveiled', 'release', 'released'],
        'leadership_change': ['ceo', 'executive', 'appoint', 'appointed', 'resign', 'resigned', 'management'],
        'regulatory': ['regulation', 'regulatory', 'compliance', 'regulator', 'sec', 'fined', 'penalty'],
        'investment': ['invest', 'investment', 'funding', 'fund', 'capital', 'venture', 'stake'],
        'restructuring': ['restructure', 'restructuring', 'layoff', 'layoffs', 'downsize', 'downsizing', 'cost-cutting'],
        'dividend': ['dividend', 'dividends', 'payout', 'yield', 'shareholder', 'stockholder'],
        'litigation': ['lawsuit', 'litigation', 'legal', 'sue', 'sued', 'court', 'settlement'],
        'market_expansion': ['expansion', 'expand', 'global', 'international', 'enter', 'entered', 'market']
    }
    
    # Industries/Sectors
    INDUSTRIES = {
        'technology': ['tech', 'software', 'hardware', 'ai', 'artificial intelligence', 'cloud', 'computing', 'semiconductor', 'internet'],
        'finance': ['bank', 'financial', 'insurance', 'fintech', 'credit', 'loan', 'mortgage', 'invest', 'asset'],
        'healthcare': ['health', 'medical', 'pharma', 'biotech', 'drug', 'therapeutics', 'hospital', 'patient'],
        'energy': ['oil', 'gas', 'energy', 'renewable', 'solar', 'wind', 'petroleum', 'power', 'utility'],
        'consumer': ['retail', 'consumer', 'goods', 'apparel', 'food', 'beverage', 'restaurant', 'e-commerce'],
        'industrial': ['manufacturing', 'industrial', 'construction', 'machinery', 'aerospace', 'defense', 'chemical'],
        'telecom': ['telecom', 'communication', 'wireless', 'broadband', 'network', 'mobile', 'phone'],
        'automotive': ['auto', 'automotive', 'vehicle', 'car', 'electric', 'ev', 'transportation', 'mobility'],
        'materials': ['materials', 'mining', 'metal', 'steel', 'chemicals', 'paper', 'packaging', 'commodity'],
        'media': ['media', 'entertainment', 'streaming', 'content', 'advertising', 'publishing', 'broadcast'],
        'realestate': ['real estate', 'property', 'reit', 'housing', 'commercial', 'residential', 'development']
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