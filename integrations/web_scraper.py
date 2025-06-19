import requests


def Webscraper(domain_url:str):

    text_response = requests.get(f"https://r.jina.ai/{domain_url}")

    return text_response