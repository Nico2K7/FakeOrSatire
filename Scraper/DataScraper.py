from requests import get
from bs4 import BeautifulSoup
import re


class Scraper():
    def __init__(self, link):
        self.url = link

    def get_parsed_text(self):
        headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"}
        response = get(self.url, headers=headers)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        for script in html_soup(["script", "style"]):
            script.decompose()  # rip it out

        all_p_elements = html_soup.findAll('p')
        text = ""
        for elem in all_p_elements:
            if elem.text == '\xa0':
                continue
            elem_elements =[]
            for elem_part in elem.findAll(text = True):
                elem_part.replace('\n','')
                elem_part.replace('\t','')
                elem_part = elem_part.strip()
                if elem_part == '':
                    continue
                elem_elements.append(elem_part)
            text += '\n' + ''.join(elem_elements)
        text = text.strip()
        reg_exp_script = re.compile('<script|</script>')
        text_no_script = re.sub(reg_exp_script, '', text)
        reg_exp = re.compile('<.*>|<.*\"')
        clean_text = re.sub(reg_exp, '', text_no_script)
        return clean_text


if __name__ == "__main__":
    url = 'https://en.wikipedia.org/wiki/W-shingling'
    scraper = Scraper(url)
    parsed_text = scraper.get_parsed_text()
    print(parsed_text)
