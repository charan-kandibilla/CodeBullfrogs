import ssl
import socket
import datetime

def check_ssl_expiry(domain):
    """
    Checks SSL certificate expiration date.

    :param domain: The domain name (without 'http' or 'https').
    :return: Dictionary with SSL status and expiry details.
    """
    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                expiry_date = datetime.datetime.strptime(cert['notAfter'], "%b %d %H:%M:%S %Y GMT")
                days_left = (expiry_date - datetime.datetime.utcnow()).days
                
                return {
                    "ssl_valid": days_left > 0,
                    "days_left": days_left,
                    "expiry_date": expiry_date.strftime("%Y-%m-%d")
                }

    except Exception as e:
        return {"error": f"SSL check failed: {str(e)}"}

