import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
from typing import Dict, List
import pandas as pd
from src.feature_extractors.extractor_base import FeatureExtractorBase


class IndustrySectorClassifier(FeatureExtractorBase):
    """
    A context-aware feature extractor for classifying financial news sentences into industry sectors.
    """
    
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
    CONTEXT_PATTERNS = {
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
    ENTITY_PATTERNS = {
        "company_name": r"([A-Z][a-z]*\.?\s)?([A-Z][a-z]+\s)*[A-Z][a-z]*\.?(\s(Inc|Corp|Co|Ltd|LLC|Group|SA|AG|SE|NV|PLC|GmbH)\.?)?",
        "ticker_symbol": r"\$[A-Z]{1,5}",
        "industry_terms": r"(sector|industry|segment|market|business)"
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize the IndustrySectorClassifier.
        
        Args:
            config (Dict): Configuration dictionary with the following possible keys:
                - input_col (str): Name of the input column containing text (default: 'Sentence')
                - output_col (str): Name of the output column for sector classification (default: 'Sector')
                - preprocess (bool): Whether to preprocess text before classification (default: True)
                - add_confidence (bool): Whether to add confidence scores as additional columns (default: False)
                - min_confidence (float): Minimum confidence threshold to classify (default: 0.25)
                - use_contextual (bool): Whether to use contextual pattern matching (default: True)
                - context_weight (float): Weight for contextual pattern match scores (default: 1.5)
                - allow_multiple (bool): Whether to return multiple sectors if confidence is close (default: False)
                - multi_threshold (float): Threshold for multiple sector selection (default: 0.85)
        """
        # Compile patterns for faster matching
        self._compile_patterns()
        
        super().__init__(config)
        
        # Set default configuration values if not provided
        self.input_col = self.config.get('input_col', 'Sentence')
        self.output_col = self.config.get('output_col', 'Sector')
        self.preprocess = self.config.get('preprocess', True)
        self.add_confidence = self.config.get('add_confidence', False)
        self.min_confidence = self.config.get('min_confidence', 0.25)
        self.use_contextual = self.config.get('use_contextual', True)
        self.context_weight = self.config.get('context_weight', 1.5)
        self.allow_multiple = self.config.get('allow_multiple', False)
        self.multi_threshold = self.config.get('multi_threshold', 0.85)
        
        # Initialize NLP components
        self._init_nlp()
        
        # Load industry-related stock tickers (optional, could be extended)
        self.industry_tickers = self._load_industry_tickers()
    
    def _compile_patterns(self):
        """Compile regex patterns for faster matching"""
        # Compile context patterns
        for sector in self.CONTEXT_PATTERNS:
            compiled_patterns = []
            for pattern, weight in self.CONTEXT_PATTERNS[sector]:
                compiled_patterns.append((re.compile(pattern, re.IGNORECASE), weight))
            self.CONTEXT_PATTERNS[sector] = compiled_patterns
            
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
            
            # Regular expressions for detecting companies and industries
            self.company_pattern = re.compile(self.ENTITY_PATTERNS["company_name"].pattern)
            self.ticker_pattern = re.compile(self.ENTITY_PATTERNS["ticker_symbol"].pattern)
    
    def _load_industry_tickers(self):
        """
        Load mapping of stock tickers to industries (placeholder method).
        In a real implementation, this would load from a database or file.
        
        Returns:
            dict: Mapping of ticker symbols to industry sectors
        """
        # This is a simplified example - in practice, this would be a much larger dataset
        return {
            "AAPL": "technology",
            "MSFT": "technology",
            "GOOGL": "technology",
            "AMZN": "technology",
            "META": "technology",
            "JPM": "finance",
            "BAC": "finance",
            "GS": "finance",
            "JNJ": "healthcare",
            "PFE": "healthcare",
            "UNH": "healthcare",
            "XOM": "energy",
            "CVX": "energy",
            "NEE": "energy",
            "WMT": "consumer_goods",
            "PG": "consumer_goods",
            "KO": "consumer_goods",
            "CAT": "manufacturing", 
            "GE": "manufacturing",
            "MMM": "manufacturing",
            "VZ": "telecommunications",
            "T": "telecommunications",
            "TMUS": "telecommunications",
            "UPS": "transportation",
            "FDX": "transportation",
            "BA": "transportation",
            "SPG": "real_estate",
            "AMT": "real_estate",
            "WELL": "real_estate",
            "DIS": "media_entertainment",
            "NFLX": "media_entertainment",
            "CMCSA": "media_entertainment",
            "DOW": "materials",
            "FCX": "materials",
            "NEM": "materials",
            "ADM": "agriculture",
            "DE": "agriculture",
            "MOS": "agriculture",
        }
    
    def preprocess_text(self, text):
        """
        Preprocess text for classification.
        
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
        
        # Extract and save company names and tickers for later use
        self.companies = self.company_pattern.findall(text)
        self.tickers = self.ticker_pattern.findall(text)
        
        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return " ".join(processed_tokens)
        
    def extract_entities(self, text):
        """
        Extract company names, ticker symbols, and industry terms from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of extracted entities
        """
        entities = {
            "companies": self.company_pattern.findall(text),
            "tickers": self.ticker_pattern.findall(text),
            "industry_terms": re.findall(self.ENTITY_PATTERNS["industry_terms"], text.lower())
        }
        
        # Process tickers to remove '$' prefix
        if entities["tickers"]:
            entities["tickers"] = [ticker[1:] for ticker in entities["tickers"]]  # Remove '$' prefix
            
            # Check if tickers map to known industries
            entities["ticker_industries"] = []
            for ticker in entities["tickers"]:
                if ticker in self.industry_tickers:
                    entities["ticker_industries"].append(
                        (ticker, self.industry_tickers[ticker])
                    )
        
        return entities
    
    def detect_contextual_patterns(self, text):
        """
        Detect contextual patterns in the text related to industries.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of sector matches and their scores
        """
        if not self.use_contextual:
            return {}
            
        scores = {}
        matches = {}
        
        # Check contextual patterns for each sector
        for sector in self.CONTEXT_PATTERNS:
            sector_score = 0
            sector_matches = []
            
            # Try each pattern
            for pattern, weight in self.CONTEXT_PATTERNS[sector]:
                pattern_matches = pattern.findall(text.lower())
                if pattern_matches:
                    sector_score += weight * len(pattern_matches)
                    sector_matches.extend(pattern_matches)
            
            if sector_score > 0:
                scores[sector] = sector_score
                matches[sector] = sector_matches
        
        return {
            "scores": scores,
            "matches": matches
        }
    
    def classify_with_keywords(self, text):
        """
        Classify text using keyword matching.
        
        Args:
            text (str): Text to classify
            
        Returns:
            tuple: (sector_scores, matched_keywords)
        """
        if not isinstance(text, str):
            return {}, {}
            
        text = text.lower()
        scores = {}
        matched_keywords = {}
        
        # Calculate scores for each sector based on keyword matches
        for sector, keywords in self.INDUSTRY_KEYWORDS.items():
            sector_score = 0
            matches = []
            
            for keyword in keywords:
                if keyword.lower() in text:
                    # Weight multi-word keywords higher
                    if " " in keyword:
                        sector_score += 1.5
                    else:
                        sector_score += 1.0
                    matches.append(keyword)
            
            if sector_score > 0:
                # Normalize by the number of keywords
                scores[sector] = sector_score / len(keywords)
                matched_keywords[sector] = matches
        
        return scores, matched_keywords
    
    def check_overlaps(self, scores):
        """
        Check for potential industry overlaps and adjust scores accordingly.
        
        Args:
            scores (dict): Sector scores
            
        Returns:
            dict: Adjusted sector scores
        """
        adjusted_scores = scores.copy()
        
        # Check each defined overlap
        for (sector1, sector2), description in self.INDUSTRY_OVERLAPS.items():
            if sector1 in scores and sector2 in scores:
                # If both sectors have good scores, give a slight boost to both
                if scores[sector1] >= 0.3 and scores[sector2] >= 0.3:
                    boost = min(scores[sector1], scores[sector2]) * 0.2
                    adjusted_scores[sector1] += boost
                    adjusted_scores[sector2] += boost
        
        return adjusted_scores
    
    def classify_text(self, text):
        """
        Classify text into an industry sector using combined keyword and contextual pattern matching.
        
        Args:
            text (str): Text to classify
            
        Returns:
            tuple: (predicted_sector, confidence, all_scores)
        """
        if not isinstance(text, str):
            return "unknown", 0.0, {}
            
        # Preprocess text
        preprocessed = self.preprocess_text(text)
        
        # Extract entities (companies, tickers, industry terms)
        entities = self.extract_entities(text)
        
        # Get scores from keyword matching
        keyword_scores, matched_keywords = self.classify_with_keywords(preprocessed)
        
        # Get scores from contextual pattern matching
        context_results = self.detect_contextual_patterns(text)
        context_scores = context_results.get("scores", {})
        
        # Check ticker-based industry hints
        ticker_boost = {}
        if "ticker_industries" in entities and entities["ticker_industries"]:
            for ticker, industry in entities["ticker_industries"]:
                if industry not in ticker_boost:
                    ticker_boost[industry] = 0
                ticker_boost[industry] += 0.5  # Boost for each ticker matching a known industry
        
        # Combine scores
        combined_scores = {}
        
        # Start with all sectors from all methods
        all_sectors = set(list(keyword_scores.keys()) + list(context_scores.keys()) + list(ticker_boost.keys()))
        
        for sector in all_sectors:
            keyword_score = keyword_scores.get(sector, 0)
            context_score = context_scores.get(sector, 0)
            ticker_score = ticker_boost.get(sector, 0)
            
            # Weight contextual patterns higher
            if self.use_contextual:
                combined_scores[sector] = keyword_score + (context_score * self.context_weight) + ticker_score
            else:
                combined_scores[sector] = keyword_score + ticker_score
        
        # Check for industry overlaps and adjust scores
        combined_scores = self.check_overlaps(combined_scores)
        
        # If no sector matched, return "unknown"
        if not combined_scores:
            return "unknown", 0.0, {}
        
        # Normalize scores
        total_score = sum(combined_scores.values())
        for sector in combined_scores:
            combined_scores[sector] /= total_score
        
        # Get the sector with the highest score
        predicted_sector = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[predicted_sector]
        
        # Apply minimum confidence threshold
        if confidence < self.min_confidence:
            return "unknown", confidence, combined_scores
        
        # Check if we should return multiple sectors
        if self.allow_multiple:
            sectors = []
            for sector, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
                if score >= confidence * self.multi_threshold:
                    sectors.append(sector)
                    # Limit to top 2 sectors
                    if len(sectors) == 2:
                        break
            
            if len(sectors) > 1:
                return "|".join(sectors), confidence, combined_scores
        
        return predicted_sector, confidence, combined_scores
    
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
        Transform the input data by adding sector classification.
        
        Args:
            X (pd.DataFrame or pd.Series or list): Input data containing text samples
            
        Returns:
            pd.DataFrame: Transformed data with sector classification
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
        
        # Get sector predictions
        sector_predictions = []
        confidence_scores = []
        sector_scores = []
        
        for text in X_text:
            sector, confidence, scores = self.classify_text(text)
            sector_predictions.append(sector)
            confidence_scores.append(confidence)
            sector_scores.append(scores)
        
        # Add the predictions as a new column
        result[self.output_col] = sector_predictions
        
        # Optionally add confidence scores
        if self.add_confidence:
            result[f"{self.output_col}_confidence"] = confidence_scores
            
            # Optionally add individual sector scores
            if self.config.get('add_sector_scores', False):
                for sector in self.INDUSTRY_SECTORS:
                    result[f"{sector}_score"] = [scores.get(sector, 0.0) for scores in sector_scores]
        
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
            
            if self.config.get('add_sector_scores', False):
                for sector in self.INDUSTRY_SECTORS:
                    feature_names.append(f"{sector}_score")
                    
        return feature_names
    
    def predict(self, texts):
        """
        Predict the industry sector for input texts.
        
        Args:
            texts (list or str or pd.Series): Input text(s) to classify
            
        Returns:
            list: Predicted industry sectors
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform the texts and extract the predictions
        result = self.transform(texts)
        
        return result[self.output_col].tolist()
    
    def explain_classification(self, text):
        """
        Explain why a text was classified in a particular sector.
        
        Args:
            text (str): Text to explain
            
        Returns:
            dict: Explanation of the classification
        """
        # Preprocess text
        preprocessed = self.preprocess_text(text)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Get scores and matched keywords
        keyword_scores, matched_keywords = self.classify_with_keywords(preprocessed)
        
        # Get context patterns
        context_results = self.detect_contextual_patterns(text)
        context_scores = context_results.get("scores", {})
        context_matches = context_results.get("matches", {})
        
        # Check ticker-based industry hints
        ticker_hints = []
        if "ticker_industries" in entities and entities["ticker_industries"]:
            ticker_hints = entities["ticker_industries"]
        
        # Get final classification
        sector, confidence, all_scores = self.classify_text(text)
        
        # Build explanation
        explanation = {
            "classification": sector,
            "confidence": confidence,
            "all_sector_scores": all_scores,
            "keyword_evidence": matched_keywords,
            "contextual_evidence": context_matches,
            "entities_detected": {
                "companies": entities.get("companies", []),
                "tickers": entities.get("tickers", []),
                "industry_terms": entities.get("industry_terms", [])
            },
            "ticker_industry_hints": ticker_hints
        }
        
        # Check for overlaps
        overlaps = []
        if "|" in sector:
            sectors = sector.split("|")
            for (s1, s2), desc in self.INDUSTRY_OVERLAPS.items():
                if s1 in sectors and s2 in sectors:
                    overlaps.append({"sectors": (s1, s2), "description": desc})
        
        if overlaps:
            explanation["industry_overlaps"] = overlaps
        
        return explanation
    
    def update_keywords(self, sector, new_keywords, replace=False):
        """
        Update the keywords for a specific sector.
        
        Args:
            sector (str): Sector to update
            new_keywords (list): New keywords to add
            replace (bool): Whether to replace existing keywords or append (default: False)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.INDUSTRY_SECTORS:
            print(f"Error: Sector '{sector}' not found")
            return False
            
        if replace:
            self.INDUSTRY_KEYWORDS[sector] = new_keywords
        else:
            # Add only unique keywords
            existing_keywords = set(self.INDUSTRY_KEYWORDS[sector])
            for keyword in new_keywords:
                if keyword not in existing_keywords:
                    self.INDUSTRY_KEYWORDS[sector].append(keyword)
        
        return True
    
    def add_context_pattern(self, sector, pattern, weight=1.0):
        """
        Add a new context pattern for a sector.
        
        Args:
            sector (str): Sector to add pattern for
            pattern (str): Regular expression pattern
            weight (float): Weight for the pattern
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.INDUSTRY_SECTORS:
            print(f"Error: Sector '{sector}' not found")
            return False
        
        # Compile pattern
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            print(f"Error: Invalid regex pattern '{pattern}'")
            return False
        
        # Initialize pattern list if needed
        if sector not in self.CONTEXT_PATTERNS:
            self.CONTEXT_PATTERNS[sector] = []
        
        # Add pattern
        self.CONTEXT_PATTERNS[sector].append((compiled_pattern, weight))
        
        return True
    
    def add_ticker_industry_mapping(self, ticker, sector):
        """
        Add or update a ticker-to-industry mapping.
        
        Args:
            ticker (str): Stock ticker symbol (without $ prefix)
            sector (str): Industry sector
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.INDUSTRY_SECTORS:
            print(f"Error: Sector '{sector}' not found")
            return False
            
        self.industry_tickers[ticker.upper()] = sector
        return True
    
    def analyze_unclassified(self, texts, threshold=0.25):
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
        sector_distribution = {}
        
        for text in texts:
            sector, confidence, scores = self.classify_text(text)
            
            if confidence < threshold:
                unclassified.append({
                    "text": text,
                    "best_sector": sector,
                    "confidence": confidence,
                    "top_scores": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                })
            elif confidence < 0.5:
                low_confidence.append({
                    "text": text,
                    "sector": sector,
                    "confidence": confidence
                })
            
            # Track sector distribution
            if sector not in sector_distribution:
                sector_distribution[sector] = 0
            sector_distribution[sector] += 1
        
        # Calculate percentage of unclassified
        unclassified_pct = (len(unclassified) / len(texts)) * 100 if texts else 0
        
        return {
            "unclassified_count": len(unclassified),
            "unclassified_percentage": unclassified_pct,
            "low_confidence_count": len(low_confidence),
            "sector_distribution": sector_distribution,
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
        
        # Collect high confidence samples for each sector
        for text in texts:
            sector, confidence, _ = self.classify_text(text)
            
            if confidence >= min_confidence:
                # Handle multi-sector cases
                if "|" in sector:
                    sectors = sector.split("|")
                    for s in sectors:
                        if s not in high_confidence_samples:
                            high_confidence_samples[s] = []
                        high_confidence_samples[s].append((text, confidence))
                else:
                    if sector not in high_confidence_samples:
                        high_confidence_samples[sector] = []
                    high_confidence_samples[sector].append((text, confidence))
        
        # Use these to improve keyword lists
        improvements = {}
        
        for sector, samples in high_confidence_samples.items():
            if len(samples) < 5:  # Need enough samples
                continue
                
            # Extract potentially useful new keywords
            all_text = " ".join([self.preprocess_text(s[0]) for s in samples])
            words = all_text.split()
            word_counts = Counter(words)
            
            # Find words that might be good indicators but aren't in keywords
            existing_keywords = set([kw.lower() for kw in self.INDUSTRY_KEYWORDS.get(sector, [])])
            
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
                added = self.update_keywords(sector, new_keywords[:5])
                improvements[sector] = new_keywords[:5]
        
        return {
            "sectors_improved": list(improvements.keys()),
            "new_keywords_added": improvements,
            "high_confidence_samples_found": {k: len(v) for k, v in high_confidence_samples.items()}
        }