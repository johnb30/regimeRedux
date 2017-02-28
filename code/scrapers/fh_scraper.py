import os
import sys
import json
import time
import requests
import lxml.html as lh

BASE = 'https://freedomhouse.org'


def get_countries(url):
    """
    Obtain the URLs for each individual country..

    Parameters
    ----------

    url: string.
            URL for the region.

    Returns
    -------

    urls: list.
            URLs for each

    """
    print('Obtaining country links...')
    r = requests.get(url)
    doc = lh.fromstring(r.content)
    links = doc.xpath('//span[@class="field-content"]//a/@href')
    urls = [BASE + l for l in links]

    return urls


def process_url_content(page):
    """
    Get the actual report text.

    Paramters
    ---------

    page: string
            URL to the country page

    Returns
    -------

    content: string
                Report content.

    title: string
            Country name for the report.
    """
    print('Obtaining content...')
    doc = lh.fromstring(requests.get(page).content)
    pars = doc.xpath('//p[@class="rtejustify"]')

    content = ''
    for p in pars:
        content += p.text_content() + '\n'

    title = doc.xpath('//h1')[0].text_content()

    time.sleep(1)

    return content, title


def process(write_dir):
    """
    Main function.

    Parameters
    ----------

    write_dir: string
                Directory to which reports will be written.
    """
    for i in xrange(1999, 2015):
        output_data = []
        print('Processing year: {}'.format(i))
        url = 'https://freedomhouse.org/report/freedom-world/freedom-world-{}'.format(i)
        country_links = get_countries(url)

        for l in country_links:
            content, country = process_url_content(l)
            country_data = {'year': i, 'url': l, 'source': 'FHpress',
                            'name': country, 'data': content}

            output_data.append(country_data)

        write_file = os.path.join(write_dir, 'FHpress_{}.json'.format(i))
        print('Writing...')
        with open(write_file, 'w') as f:
            json.dump(output_data, f)

if __name__ == '__main__':
    write_dir = sys.argv[1]
    print('Running...')
    process(write_dir)
