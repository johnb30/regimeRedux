import os
import sys
import json
import time
import requests
import lxml.html as lh

BASE = 'https://www.state.gov'


def get_menu_links(url):
    """
    Each report has a navigation link on the side. This contains
    the regional divisions such as "Africa." These regional
    pages lead to individual country pages obtained in `get_countries`.

    Parameters
    ----------

    url: string.
            URL for a given year's set of reporting.

    Returns
    -------

    links: list.
            URLs for each regional division.
    """
    page = requests.get(url).content
    doc = lh.fromstring(page)

    links = doc.xpath('//ul[@class="menu"]//a')

    not_phrase = ['Front Matter', 'Appendices', 'Related Material',
                  'Preface, Overview and Acknowledgements, and Introduction',
                  'Appendixes']

    links = [l.attrib['href'] for l in links if l.attrib['title']
             not in not_phrase]

    return links


def get_countries(url, yr):
    """
    Obtain the URLs for each individual country in a given region
    for a given year.

    Parameters
    ----------

    url: string.
            URL for the region.

    yr: int.
        Year being scraped.

    Returns
    -------

    urls: list.
            URLs for each

    """
    print('Obtaining country links...')
    r = requests.get(url)
    doc = lh.fromstring(r.content)
    links = doc.xpath('//div[@class="l-wrap"]//a/@href')

    urls = []
    for x in links:
        if '{}'.format(yr) in x:
            urls.append(BASE + x)

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
    content = doc.xpath('//div[@id="centerblock"]')[0].text_content()

    title = doc.xpath('//h2[@id="page-title"]')[0].text_content()

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
    for i in xrange(1999, 2016):
        output_data = []
        print('Processing year: {}'.format(i))
        url = 'https://www.state.gov/j/drl/rls/hrrpt/{}/index.htm'.format(i)
        sector_links = get_menu_links(url)

        for l in sector_links:
            print('Processing sector: {}'.format(l))
            country_urls = get_countries(l, i)

            for u in country_urls:
                content, country = process_url_content(u)
                country_data = {'year': i, 'url': u, 'source': 'StateHR',
                                'name': country, 'data': content}
                output_data.append(country_data)

        write_file = os.path.join(write_dir, 'StateHR_{}.json'.format(i))
        print('Writing...')
        with open(write_file, 'w') as f:
            json.dump(output_data, f)


if __name__ == '__main__':
    write_dir = sys.argv[1]
    print('Running...')
    process(write_dir)
