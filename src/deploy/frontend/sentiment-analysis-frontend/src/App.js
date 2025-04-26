import React, { useState, useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import './App.css';
import Lottie from 'react-lottie';
import AOS from 'aos';
import 'aos/dist/aos.css';
import ParticlesBg from 'particles-bg';

// Import Lottie animations
import animationData from './lotties/sentiment-analysis.json';
import loadingAnimation from './lotties/loading.json';
import positiveAnimation from './lotties/positive.json';
import neutralAnimation from './lotties/neutral.json';
import negativeAnimation from './lotties/negative.json';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showHero, setShowHero] = useState(true);
  const [hasScrolled, setHasScrolled] = useState(false);
  
  const analyzeRef = useRef(null);
  const chartRef = useRef(null);
  const loaderRef = useRef(null);

    // Add these new state variables at the beginning of your App component
  const [menuOpen, setMenuOpen] = useState(false);
  const [isBackdropVisible, setIsBackdropVisible] = useState(false);
  // Add this state to track active section
  const [activeSection, setActiveSection] = useState('hero');

  // Add this useEffect to detect which section is in view
  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 100;
      
      // Get positions of different sections
      const heroSection = document.querySelector('.hero-section')?.offsetTop || 0;
      const featuresSection = document.getElementById('features')?.offsetTop || 0;
      const analyzeSection = document.getElementById('analyze')?.offsetTop || 0;
      const aboutSection = document.getElementById('about')?.offsetTop || 0;
      
      // Determine which section is currently in view
      if (scrollPosition < featuresSection) {
        setActiveSection('hero');
      } else if (scrollPosition < analyzeSection) {
        setActiveSection('features');
      } else if (scrollPosition < aboutSection) {
        setActiveSection('analyze');
      } else {
        setActiveSection('about');
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Add this function to toggle the mobile menu
  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
    setIsBackdropVisible(!menuOpen);
    
    // Prevent scrolling when menu is open
    document.body.style.overflow = !menuOpen ? 'hidden' : '';
  };

  {isBackdropVisible && (
    <div 
      className="mobile-backdrop" 
      onClick={() => {
        setMenuOpen(false);
        setIsBackdropVisible(false);
        document.body.style.overflow = '';
      }}
    />
  )}

  // Add this function to close the menu when a link is clicked
  const closeMenu = () => {
    setMenuOpen(false);
  };
    
  // Initialize AOS animations
  useEffect(() => {
    AOS.init({
      duration: 1000,
      once: false,
      mirror: false,
    });
  }, []);
  
  // Handle scroll events for navbar
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      setHasScrolled(scrollTop > 100);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  const steps = [
    { id: 1, name: 'Preprocessing text data', icon: 'fa-solid fa-file-lines' },
    { id: 2, name: 'Tokenizing and encoding', icon: 'fa-solid fa-code' },
    { id: 3, name: 'Generating inference', icon: 'fa-solid fa-brain' },
    { id: 4, name: 'Processing results', icon: 'fa-solid fa-chart-simple' }
  ];
  
  useEffect(() => {
    let timer;
    if (isAnalyzing && currentStep < steps.length) {
      timer = setTimeout(() => {
        // Make the actual API call
        if (currentStep == 0) {
          fetchSentimentAnalysis();
        }
        setCurrentStep(prevStep => prevStep + 1);
        
        
      }, 3000);
    }
    
    return () => clearTimeout(timer);
  }, [isAnalyzing, currentStep]);
  
  const scrollToAnalyze = () => {
    setShowHero(false);
    setTimeout(() => {
      if (analyzeRef.current) {
        analyzeRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    
    setIsAnalyzing(true);
    setCurrentStep(0);
    setResult(null);
    setError(null);
    
    // Scroll to the loading section
    setTimeout(() => {
      if (loaderRef.current) {
        loaderRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };
  
  const fetchSentimentAnalysis = async () => {
    try {
      const response = await fetch('https://sentiment-analysis-service-181836234900.us-central1.run.app/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to analyze sentiment');
      }
      
      const data = await response.json();
      setResult(data.predictions[0]);
      setIsAnalyzing(false);
      
      // Scroll to results
      if (chartRef.current) {
        setTimeout(() => {
          chartRef.current.scrollIntoView({ behavior: 'smooth' });
        }, 500);
      }
    } catch (err) {
      setError(err.message);
      setIsAnalyzing(false);
    }
  };
  
  const getSentimentAnimation = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return positiveAnimation;
      case 'neutral':
        return neutralAnimation;
      case 'negative':
        return negativeAnimation;
      default:
        return positiveAnimation;
    }
  };
  
  const getSentimentTitle = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'Positive Sentiment';
      case 'neutral':
        return 'Neutral Sentiment';
      case 'negative':
        return 'Negative Sentiment';
      default:
        return 'Unknown Sentiment';
    }
  };
  
  const getLottieOptions = (animData, loop = true) => {
    return {
      loop: loop,
      autoplay: true,
      animationData: animData,
      rendererSettings: {
        preserveAspectRatio: 'xMidYMid slice'
      }
    };
  };
  
  const getChartData = () => {
    if (!result) return {
      labels: ['Negative', 'Neutral', 'Positive'],
      datasets: [
        {
          label: 'Sentiment Probability',
          data: [0, 0, 0],
          backgroundColor: [
            'rgba(239, 68, 68, 0.7)',
            'rgba(100, 116, 139, 0.7)',
            'rgba(34, 197, 94, 0.7)',
          ],
          borderColor: [
            'rgb(239, 68, 68)',
            'rgb(100, 116, 139)',
            'rgb(34, 197, 94)',
          ],
          borderWidth: 1,
        },
      ],
    };
    
    return {
      labels: ['Negative', 'Neutral', 'Positive'],
      datasets: [
        {
          label: 'Sentiment Probability',
          data: result.raw_predictions,
          backgroundColor: [
            'rgba(239, 68, 68, 0.7)',
            'rgba(100, 116, 139, 0.7)',
            'rgba(34, 197, 94, 0.7)',
          ],
          borderColor: [
            'rgb(239, 68, 68)',
            'rgb(100, 116, 139)',
            'rgb(34, 197, 94)',
          ],
          borderWidth: 1,
        },
      ],
    };
  };
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Sentiment Analysis Probabilities',
        font: {
          size: 16,
          family: "'Inter', sans-serif",
        },
        padding: {
          top: 10,
          bottom: 20
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const value = context.raw;
            const percentage = (value * 100).toFixed(2) + '%';
            return percentage;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          callback: function(value) {
            return (value * 100) + '%';
          },
          font: {
            family: "'Inter', sans-serif",
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      x: {
        ticks: {
          font: {
            family: "'Inter', sans-serif",
          }
        },
        grid: {
          display: false
        }
      }
    },
    animation: {
      duration: 2000,
      easing: 'easeOutQuart'
    },
  };
  
  return (
    <>
      <ParticlesBg type="cobweb" bg={true} color="#6366f1" num={80} />
      <nav className={`navbar ${hasScrolled ? 'navbar-scrolled' : ''}`}>
        <div className="nav-container">
          <div className="nav-logo">
            <i className="fa-solid fa-brain"></i>
            <span>SentimentAI</span>
          </div>
          
          <div className="hamburger" onClick={toggleMenu}>
            <i className={`fa-solid ${menuOpen ? 'fa-xmark' : 'fa-bars'}`}></i>
          </div>    

          {menuOpen && (
            <div 
              className="mobile-backdrop" 
              onClick={() => {
                setMenuOpen(false);
                document.body.style.overflow = '';
              }}
            />
          )}
        
          <div className={`nav-links ${menuOpen ? 'nav-active' : ''}`}>
          <a href="#features" className={`nav-link ${activeSection === 'features' ? 'active' : ''}`} onClick={closeMenu}>Features</a>
          <a href="#analyze" className={`nav-link ${activeSection === 'analyze' ? 'active' : ''}`} onClick={closeMenu}>Try It</a>
          <a href="#about" className={`nav-link ${activeSection === 'about' ? 'active' : ''}`} onClick={closeMenu}>About</a>
            <button className="nav-button" onClick={() => { scrollToAnalyze(); closeMenu(); }}>
              Analyze Now
              <i className="fa-solid fa-arrow-right"></i>
            </button>
          </div>
        </div>
      </nav>
      
      {showHero && (
        <div className="hero-section">
          <div className="hero-content">
            <h1 data-aos="fade-up">Sentiment Analysis <span className="gradient-text">Reimagined</span></h1>
            <p data-aos="fade-up" data-aos-delay="100">
              Harness the power of advanced AI to analyze sentiment in any text.
              Get instant, accurate insights with our state-of-the-art machine learning model.
            </p>
            <div className="hero-buttons" data-aos="fade-up" data-aos-delay="200">
              <button className="hero-button primary" onClick={scrollToAnalyze}>
                Try It Now
                <i className="fa-solid fa-rocket"></i>
              </button>
              <a href="#features" className="hero-button secondary">
                Explore Features
                <i className="fa-solid fa-arrow-right"></i>
              </a>
            </div>
          </div>
          <div className="hero-animation" data-aos="zoom-in" data-aos-delay="300">
            <Lottie options={getLottieOptions(animationData)} height={400} width={400} />
          </div>
        </div>
      )}
      
      <div className="stats-banner">
        <div className="stat-item" data-aos="fade-up" data-aos-delay="100">
          <div className="stat-number">99.4%</div>
          <div className="stat-label">Accuracy</div>
        </div>
        <div className="stat-item" data-aos="fade-up" data-aos-delay="200">
          <div className="stat-number">250ms</div>
          <div className="stat-label">Response Time</div>
        </div>
        <div className="stat-item" data-aos="fade-up" data-aos-delay="300">
          <div className="stat-number">3</div>
          <div className="stat-label">Sentiment Classes</div>
        </div>
        <div className="stat-item" data-aos="fade-up" data-aos-delay="400">
          <div className="stat-number">10M+</div>
          <div className="stat-label">Analyses Run</div>
        </div>
      </div>
      
      <div className="container">
        <section id="analyze" className="card analysis-card" ref={analyzeRef} data-aos="fade-up">
          <div className="card-header">
            <h2 className="card-title">
              <i className="fa-solid fa-magnifying-glass"></i>
              Text Analysis
            </h2>
            <p className="card-subtitle">Enter any text below to analyze its sentiment</p>
          </div>
          <form onSubmit={handleSubmit} className="input-area">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to analyze sentiment... (e.g., 'The product exceeded my expectations and I'm very satisfied with my purchase.')"
              disabled={isAnalyzing}
              required
            />
            <div className="button-container">
              <button type="submit" className="btn btn-primary" disabled={isAnalyzing}>
                <i className="fa-solid fa-bolt"></i>
                Analyze Sentiment
              </button>
            </div>
          </form>
        </section>
        
        {isAnalyzing && (
          <section className="card" ref={loaderRef} data-aos="fade-up">
            <div className="loader-container">
              <div className="loader-animation">
                <Lottie options={getLottieOptions(loadingAnimation)} height={150} width={150} />
              </div>
              <div className="loader-steps">
                {steps.map((step, index) => (
                  <div
                    key={step.id}
                    className={`loader-step ${currentStep >= index ? 'active' : ''} ${currentStep > index ? 'completed' : ''}`}
                  >
                    <i className={step.icon}></i>
                    <span>{step.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
        
        {error && (
          <section className="card error-card" data-aos="fade-up">
            <div className="error-container">
              <i className="fa-solid fa-circle-exclamation"></i>
              <p>Error: {error}</p>
            </div>
          </section>
        )}
        
        {result && (
          <section className="card results-container" ref={chartRef} data-aos="fade-up">
            <div className={`sentiment-result result-${result.sentiment}`}>
              <div className="result-animation">
                <Lottie 
                  options={getLottieOptions(getSentimentAnimation(result.sentiment), false)} 
                  height={120} 
                  width={120} 
                />
              </div>
              <div className="sentiment-content">
                <h3>{getSentimentTitle(result.sentiment)}</h3>
                <p className="analysis-text">"{text}"</p>
                <p className="confidence-text">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
                <div className="confidence-bar">
                  <div 
                    className={`confidence-fill confidence-fill-${result.sentiment}`} 
                    style={{ 
                      '--final-width': `${result.confidence * 100}%`,
                      width: `${result.confidence * 100}%` 
                    }}
                  ></div>
                </div>
              </div>
            </div>
            
            <div className="chart-container">
              <h3 className="chart-title">Sentiment Breakdown</h3>
              <Bar data={getChartData()} options={chartOptions} />
            </div>
          </section>
        )}
        
        <section id="features" className="features-section" data-aos="fade-up">
          <h2 className="section-title">
            <i className="fa-solid fa-star"></i>
            Advanced Sentiment Analysis Features
          </h2>
          <p className="section-subtitle">
            Our model provides cutting-edge sentiment analysis capabilities powered by the latest advancements in AI
          </p>
          
          <div className="features-grid">
            <div className="feature-card" data-aos="fade-up" data-aos-delay="100">
              <div className="feature-icon">
                <i className="fa-solid fa-gauge-high"></i>
              </div>
              <h3 className="feature-title">High Accuracy</h3>
              <p className="feature-description">
                Our model achieves exceptional accuracy rates on benchmark datasets, consistently outperforming traditional sentiment analysis approaches.
              </p>
            </div>
            
            <div className="feature-card" data-aos="fade-up" data-aos-delay="200">
              <div className="feature-icon">
                <i className="fa-solid fa-language"></i>
              </div>
              <h3 className="feature-title">Context Understanding</h3>
              <p className="feature-description">
                Advanced natural language processing capabilities allow the model to understand context, nuance, and linguistic subtleties.
              </p>
            </div>
            
            <div className="feature-card" data-aos="fade-up" data-aos-delay="300">
              <div className="feature-icon">
                <i className="fa-solid fa-bolt-lightning"></i>
              </div>
              <h3 className="feature-title">Real-Time Analysis</h3>
              <p className="feature-description">
                Optimized for speed, our sentiment analysis model delivers results within milliseconds, making it perfect for real-time applications.
              </p>
            </div>
            
            <div className="feature-card" data-aos="fade-up" data-aos-delay="400">
              <div className="feature-icon">
                <i className="fa-solid fa-chart-line"></i>
              </div>
              <h3 className="feature-title">Confidence Scores</h3>
              <p className="feature-description">
                Get detailed confidence scores for each sentiment class, allowing for more nuanced interpretation of results.
              </p>
            </div>
            
            <div className="feature-card" data-aos="fade-up" data-aos-delay="500">
              <div className="feature-icon">
                <i className="fa-solid fa-robot"></i>
              </div>
              <h3 className="feature-title">Advanced ML Architecture</h3>
              <p className="feature-description">
                Built on state-of-the-art deep learning architecture with fine-tuning on diverse datasets for robust performance.
              </p>
            </div>
            
            <div className="feature-card" data-aos="fade-up" data-aos-delay="600">
              <div className="feature-icon">
                <i className="fa-solid fa-plug"></i>
              </div>
              <h3 className="feature-title">Easy Integration</h3>
              <p className="feature-description">
                Simple REST API makes it easy to integrate sentiment analysis capabilities into any application or workflow.
              </p>
            </div>
          </div>
        </section>
        
        <section id="how-it-works" className="how-it-works" data-aos="fade-up">
          <h2 className="section-title">
            <i className="fa-solid fa-diagram-project"></i>
            How It Works
          </h2>
          <div className="steps-container">
            <div className="step" data-aos="fade-right" data-aos-delay="100">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Input Text</h3>
                <p>Enter any text you want to analyze for sentiment, from customer reviews to social media posts.</p>
              </div>
            </div>
            <div className="step" data-aos="fade-right" data-aos-delay="200">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Advanced Processing</h3>
                <p>Our model processes the text using advanced NLP techniques, including tokenization and encoding.</p>
              </div>
            </div>
            <div className="step" data-aos="fade-right" data-aos-delay="300">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>AI Analysis</h3>
                <p>Our deep learning model analyzes the text to determine sentiment polarity and confidence scores.</p>
              </div>
            </div>
            <div className="step" data-aos="fade-right" data-aos-delay="400">
              <div className="step-number">4</div>
              <div className="step-content">
                <h3>Receive Results</h3>
                <p>Get detailed results showing sentiment classification, confidence scores, and probability breakdown.</p>
              </div>
            </div>
          </div>
        </section>
        
        <section id="use-cases" className="use-cases" data-aos="fade-up">
          <h2 className="section-title">
            <i className="fa-solid fa-lightbulb"></i>
            Use Cases
          </h2>
          <div className="use-cases-grid">
            <div className="use-case" data-aos="zoom-in" data-aos-delay="100">
              <div className="use-case-icon">
                <i className="fa-solid fa-comments"></i>
              </div>
              <h3>Customer Feedback</h3>
              <p>Analyze customer reviews and feedback to understand satisfaction levels and identify areas for improvement.</p>
            </div>
            <div className="use-case" data-aos="zoom-in" data-aos-delay="200">
              <div className="use-case-icon">
                <i className="fa-solid fa-newspaper"></i>
              </div>
              <h3>Media Monitoring</h3>
              <p>Track sentiment in news articles and social media to gauge public perception of your brand or product.</p>
            </div>
            <div className="use-case" data-aos="zoom-in" data-aos-delay="300">
              <div className="use-case-icon">
                <i className="fa-solid fa-chart-pie"></i>
              </div>
              <h3>Market Research</h3>
              <p>Understand market trends and consumer opinions by analyzing large volumes of text data quickly.</p>
            </div>
            <div className="use-case" data-aos="zoom-in" data-aos-delay="400">
              <div className="use-case-icon">
                <i className="fa-solid fa-headset"></i>
              </div>
              <h3>Support Interactions</h3>
              <p>Monitor customer support conversations to identify and prioritize negative interactions that need attention.</p>
            </div>
          </div>
        </section>
        
        <section id="about" className="about-section" data-aos="fade-up">
          <div className="about-content">
            <h2 className="section-title">About Our Technology</h2>
            <p>
              Our sentiment analysis model is built on a state-of-the-art deep learning architecture, trained on millions of labeled text examples across diverse domains. The model uses advanced natural language processing techniques to understand context, detect subtle linguistic cues, and accurately classify sentiment.
            </p>
            <p>
              The API delivers fast, reliable results with detailed confidence scores, making it perfect for integration into applications, dashboards, or automated workflows.
            </p>
            <div className="tech-stack">
              <span className="tech-badge">Deep Learning</span>
              <span className="tech-badge">NLP</span>
              <span className="tech-badge">Transfer Learning</span>
              <span className="tech-badge">REST API</span>
              <span className="tech-badge">Cloud Hosted</span>
            </div>
          </div>
        </section>
        
        <section className="cta-section" data-aos="fade-up">
          <div className="cta-content">
            <h2>Ready to analyze your text?</h2>
            <p>Try our sentiment analysis tool now and get instant insights.</p>
            <button className="cta-button" onClick={scrollToAnalyze}>
              Start Analyzing
              <i className="fa-solid fa-arrow-right"></i>
            </button>
          </div>
        </section>
      </div>
      
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <div className="footer-logo">
              <i className="fa-solid fa-brain"></i>
              <span>SentimentAI</span>
            </div>
            <p>Advanced sentiment analysis powered by cutting-edge AI technology.</p>
          </div>
          
          <div className="footer-links">
            <div className="footer-links-column">
              <h3>Quick Links</h3>
              <a href="#features">Features</a>
              <a href="#how-it-works">How It Works</a>
              <a href="#use-cases">Use Cases</a>
              <a href="#about">About</a>
            </div>
            
            <div className="footer-links-column">
              <h3>Resources</h3>
              <a href="#">Documentation</a>
              <a href="#">API Reference</a>
              <a href="#">Tutorials</a>
              <a href="#">Blog</a>
            </div>
            
            <div className="footer-links-column">
              <h3>Contact</h3>
              <a href="mailto:your.email@example.com">
                <i className="fa-solid fa-envelope"></i>
                your.email@example.com
              </a>
              <a href="https://github.com/yourusername" target="_blank" rel="noopener noreferrer">
                <i className="fa-brands fa-github"></i>
                GitHub
              </a>
              <a href="https://linkedin.com/in/yourusername" target="_blank" rel="noopener noreferrer">
                <i className="fa-brands fa-linkedin-in"></i>
                LinkedIn
              </a>
              <a href="#">
                <i className="fa-solid fa-phone"></i>
                +1 (555) 123-4567
              </a>
            </div>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p className="copyright">© {new Date().getFullYear()} SentimentAI. All rights reserved.</p>
          <div className="footer-social">
            <a href="https://github.com/yourusername" className="social-link" target="_blank" rel="noopener noreferrer">
              <i className="fa-brands fa-github"></i>
            </a>
            <a href="https://linkedin.com/in/yourusername" className="social-link" target="_blank" rel="noopener noreferrer">
              <i className="fa-brands fa-linkedin-in"></i>
            </a>
            <a href="#" className="social-link">
              <i className="fa-brands fa-twitter"></i>
            </a>
          </div>
        </div>
      </footer>
    </>
  );
}

export default App;