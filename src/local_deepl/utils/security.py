import ipaddress
import os
import socket
from urllib.parse import urlparse


def _local_ssrf_allowed() -> bool:
    return os.getenv("ALLOW_SSRF_LOCAL", "").strip().lower() == "true"


def is_ssrf_target(url: str | None) -> bool:
    """
    Validates if a URL host corresponds to a private, loopback, or internal IP address.
    Supports dynamic DNS resolution to prevent DNS rebinding attacks.

    Returns True for malformed URLs, unsupported schemes, and DNS resolution
    failures so caller-supplied endpoints fail closed.
    """
    if not url:
        return True

    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return True
        host = (parsed.hostname or "").strip().lower()
        if not host:
            return True

        allow_local = _local_ssrf_allowed()

        # Block cloud metadata endpoints regardless of local-development mode.
        if host == "metadata.google.internal":
            return True

        if host == "localhost" or host.endswith(".local"):
            return not allow_local

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None

        if ip is not None:
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return not allow_local
            return False

        addr_info = socket.getaddrinfo(host, None)
        if not addr_info:
            return True

        for _family, _socktype, _proto, _canonname, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return not allow_local

        return False
    except Exception:
        return True
