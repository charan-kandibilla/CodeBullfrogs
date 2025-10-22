from urllib.parse import urlparse
import re

def extract_url_features(url):
    features = {}
    parsed_url = urlparse(url)
    
    # Domain features
    features['domain_length'] = len(parsed_url.netloc)
    features['has_ip'] = bool(re.match(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc))
    features['special_chars'] = len(re.findall(r'[^\w\-\.]', parsed_url.netloc))
    
    return features