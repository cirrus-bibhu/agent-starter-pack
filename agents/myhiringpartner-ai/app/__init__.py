"""
MyHiringPartner AI application package
"""
# Version of the myhiringpartner-ai package
__version__ = "0.1.0"

from .agent import root_agent, EmailData, MainAgent

__all__ = ["root_agent", "EmailData", "MainAgent"]
