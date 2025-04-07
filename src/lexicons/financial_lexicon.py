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