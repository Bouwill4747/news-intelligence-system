"""
Delivery module for Telegram and Email
"""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx
import structlog
import os
from dotenv import load_dotenv
from datetime import datetime

logger = structlog.get_logger()

# Load environment variables
load_dotenv()


async def send_telegram(digest_text: str) -> bool:
    """
    Send digest to Telegram.
    Splits messages if over 4096 char limit.
    
    Args:
        digest_text: Plain text digest
    
    Returns:
        True if successful, False otherwise
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.error("telegram_config_missing")
        return False
    
    try:
        # Telegram has 4096 char limit per message
        # Split digest into chunks by story sections
        max_length = 4000
        
        if len(digest_text) <= max_length:
            messages = [digest_text]
        else:
            messages = []
            current = ""
            for line in digest_text.split('\n'):
                if len(current) + len(line) > max_length:
                    messages.append(current)
                    current = line + '\n'
                else:
                    current += line + '\n'
            if current:
                messages.append(current)
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        async with httpx.AsyncClient(timeout=30) as client:
            for i, msg in enumerate(messages):
                payload = {
                    "chat_id": chat_id,
                    "text": msg,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                }
                
                logger.info("sending_telegram_message",
                           part=i+1,
                           total_parts=len(messages))
                
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                # Small delay between messages
                if i < len(messages) - 1:
                    await asyncio.sleep(1)
        
        logger.info("telegram_sent_successfully", parts=len(messages))
        return True
    
    except Exception as e:
        logger.error("telegram_send_failed", error=str(e))
        return False


def send_email(digest_text: str) -> bool:
    """
    Send digest via email with HTML formatting.
    
    Args:
        digest_text: Plain text digest
    
    Returns:
        True if successful, False otherwise
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("EMAIL_FROM")
    to_email = os.getenv("EMAIL_TO")
    
    if not all([smtp_host, smtp_user, smtp_password, from_email, to_email]):
        logger.error("email_config_missing")
        return False
    
    try:
        subject = f"📰 Daily Intelligence Digest - {datetime.now().strftime('%B %d, %Y')}"
        
        # Create HTML version
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    max-width: 800px;
                    margin: 20px auto;
                    color: #2c3e50;
                    background-color: #f4f6f7;
                    padding: 20px;
                }}
                .container {{
                    background: #ffffff;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #ffffff;
                    padding: 8px 12px;
                    border-radius: 4px;
                    margin-top: 30px;
                }}
                .cyber-header {{ background-color: #e74c3c; }}
                .tech-header {{ background-color: #3498db; }}
                .crypto-header {{ background-color: #f39c12; }}
                .story {{
                    margin: 15px 0;
                    padding: 15px;
                    border-radius: 6px;
                    background: #f8f9fa;
                    border-left: 4px solid #3498db;
                }}
                .story-cyber {{ border-left-color: #e74c3c; }}
                .story-tech {{ border-left-color: #3498db; }}
                .story-crypto {{ border-left-color: #f39c12; }}
                .topic {{
                    font-size: 1.1em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .score {{
                    display: inline-block;
                    background: #e74c3c;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 0.85em;
                    font-weight: bold;
                }}
                .meta {{
                    color: #7f8c8d;
                    font-size: 0.85em;
                    margin: 5px 0;
                }}
                .summary {{
                    margin-top: 10px;
                    line-height: 1.6;
                }}
                .reasoning {{
                    font-style: italic;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 8px;
                }}
                .footer {{
                    text-align: center;
                    color: #95a5a6;
                    font-size: 0.8em;
                    margin-top: 30px;
                    padding-top: 15px;
                    border-top: 1px solid #ecf0f1;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌐 Daily Intelligence Digest</h1>
                <p style="text-align:center; color:#7f8c8d;">
                    {datetime.now().strftime('%B %d, %Y - %H:%M UTC')}
                </p>
        """
        
        # Parse digest text and build HTML
        current_category = None
        
        for line in digest_text.split('\n'):
            # Detect category headers
            if '🔐 CYBERSECURITY' in line:
                current_category = 'cyber'
                html += '<h2 class="cyber-header">🔐 CYBERSECURITY</h2>'
            elif '💻 TECHNOLOGY' in line:
                current_category = 'tech'
                html += '<h2 class="tech-header">💻 TECHNOLOGY</h2>'
            elif '₿ CRYPTO' in line:
                current_category = 'crypto'
                html += '<h2 class="crypto-header">₿ CRYPTO</h2>'
            
            # Detect story sections
            elif '📰 Story #' in line:
                html += f'<div class="story story-{current_category}">'
            elif line.strip().startswith('Topic:'):
                topic = line.strip().replace('Topic:', '').strip()
                html += f'<div class="topic">{topic}</div>'
            elif line.strip().startswith('Importance:'):
                score = line.strip().replace('Importance:', '').strip()
                html += f'<span class="score">⭐ {score}</span>'
            elif line.strip().startswith('Sources:'):
                sources = line.strip()
                html += f'<div class="meta">📊 {sources}</div>'
            elif line.strip().startswith('Published:'):
                published = line.strip()
                html += f'<div class="meta">📅 {published}</div>'
            elif line.strip().startswith('Summary:'):
                html += '<div class="summary">'
            elif line.strip().startswith('Why important:'):
                reasoning = line.strip().replace('Why important:', '').strip()
                html += f'</div><div class="reasoning">💡 Why important: {reasoning}</div></div>'
        
        # Close HTML
        html += f"""
                <div class="footer">
                    <p>Generated by News Intelligence System | Local LLM Powered</p>
                    <p>Sources: 18 RSS feeds | Models: Llama 3.1 8B + Llama 3.2 3B</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        
        # Attach HTML
        msg.attach(MIMEText(html, 'html'))
        
        # Send
        logger.info("sending_email", to=to_email)
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info("email_sent_successfully", to=to_email)
        return True
    
    except Exception as e:
        logger.error("email_send_failed", error=str(e))
        return False


async def deliver_digest(digest_text: str) -> dict:
    """
    Main delivery function - sends to both Telegram and Email.
    
    Args:
        digest_text: The formatted digest text
    
    Returns:
        Dict with delivery results
    """
    logger.info("starting_delivery")
    
    results = {
        'telegram': False,
        'email': False
    }
    
    # Send Telegram
    results['telegram'] = await send_telegram(digest_text)
    
    # Send Email (sync function, run in thread)
    results['email'] = await asyncio.get_event_loop().run_in_executor(
        None, send_email, digest_text
    )
    
    # Log results
    logger.info("delivery_complete", results=results)
    
    return results
