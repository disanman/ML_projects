#          _____ _            _       ____
#         |  ___(_)_ ____   _(_)____ / ___|  ___ _ __ __ _ _ __   ___ _ __
#         | |_  | | '_ \ \ / / |_  / \___ \ / __| '__/ _` | '_ \ / _ \ '__|
#         |  _| | | | | \ V /| |/ /   ___) | (__| | | (_| | |_) |  __/ |
#         |_|   |_|_| |_|\_/ |_/___| |____/ \___|_|  \__,_| .__/ \___|_|
#                                                         |_|
import requests
import lxml.html as lh   # html parser
import pandas as pd
from collections import defaultdict


class FinvizSpider():

    def __init__(self):
        self.base_link = 'https://finviz.com/screener.ashx?v=152&ft=4&o=ticker'
        self.data_fields = '&c=' + ','.join(str(x) for x in range(71))
        self.link = self.base_link + self.data_fields
        self._initialize_session()
        self.download_attemps = 5
        print('Initializing spider...')
        self._get_first_page()
        self._print_messages()

    def _initialize_session(self):
        self.session = requests.session()
        self.session.headers['user-agent'] = ('Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:60.0) '
                                              'Gecko/20100101 Firefox/60.0')
        self.session.headers['Referer'] = 'https://www.finviz.com/'

    def _get_link(self, link):
        for attemp in range(self.download_attemps):
            response = self.session.get(link)
            if response.status_code == 200:
                page = response.content
                tree = lh.fromstring(page)
                return tree
            else:
                print('Download error')

    def _parse_data_from_tree(self, tree):
        rows = tree.xpath("//tr[contains(@class, 'row-cp')]")
        for row in rows:
            row_data = row.xpath(".//text()")
            for key, value in zip(self.data_titles, row_data[1:]):
                self.data[key].append(value)  # save value in row idx of field key

    def _get_first_page(self):
        print('\nGetting data from page {1}...')
        self.page_counter = 1
        tree = self._get_link(self.link)
        # Get total number of tickers
        x = tree.xpath("//td[@class='count-text'][1]/text()")
        self.num_tickers = int(x[0].replace(' #1', ''))
        # Initialize data container: dictionary indexed by title with lists of size num_tickers (memory pre-allocation)
        self.data_titles = tree.xpath("//td[contains(@class, 'table-top')]/text()")
        self.data = defaultdict(list)
        # Get number of rows in the first page
        rows = tree.xpath("//tr[contains(@class, 'row-cp')]")
        self.rows_per_page = len(rows)
        self.num_pages_to_download = self.num_tickers // self.rows_per_page
        self.num_pages_to_download += 1 if (self.num_tickers % self.rows_per_page > 0) else 0
        # Initialize data counter
        self.data_counter = -1  # it will be increased to 0
        self._parse_data_from_tree(tree)

    def scrape_data(self, max_pages='all'):
        pages = self.num_pages_to_download if (max_pages == 'all') else int(max_pages)
        for page_number in range(1, pages + 1):
            print(f'Getting data from page {page_number}/{pages}')
            link = self.base_link + '&r=' + str(page_number * self.rows_per_page + 1)  # r=: from which row download
            tree = self._get_link(link)
            self._parse_data_from_tree(tree)
        print('Finished downloading data')

    def _print_messages(self):
        print(f'Number of tickers found: {self.num_tickers}')
        print(f'Number of pages to download: {self.num_pages_to_download}')
        print(f'Number of data fields per page: {len(self.data)}\n')

    def _clean_numeric_column(self, column_in):
        column = column_in.copy()
        is_percentage = False
        for idx, value in enumerate(column):
            if (value is None) or (value == '-') or (value == ''):
                column[idx] = None
                continue
            value = value.replace(',', '')  # decimals use .
            if value.endswith('%'):
                is_percentage = True
                column[idx] = float(value.replace('%', ''))
            elif value.endswith('B'):
                column[idx] = float(value.replace('B', '')) * 1e12
            elif value.endswith('M'):
                column[idx] = float(value.replace('M', '')) * 1e6
            elif value.endswith('K'):
                column[idx] = float(value.replace('K', '')) * 1e3
            elif ',' in value:
                column[idx] = float(value.replace(',', ''))
            else:
                column[idx] = float(value)
        return column, is_percentage

    def clean_data(self):
        self.cleaned_data = self.data.copy()
        # Categorical data
        self.categorical_cols = {'No.', 'Ticker', 'Company', 'Sector', 'Industry', 'Country', 'Earnings', 'IPO Date'}
        # Numerical_data
        self.numerical_cols = set(self.data.keys()).difference(self.categorical_cols)
        self.percentage_cols = set()
        for _, col in enumerate(self.numerical_cols):
            cleaned_list, is_percentage = self._clean_numeric_column(self.data[col])
            self.cleaned_data[col] = cleaned_list
            if is_percentage:
                self.percentage_cols.add(col)
        self.cleaned_data_df = pd.DataFrame(self.cleaned_data)
        self.cleaned_data_df[list(self.numerical_cols)].astype('float')
        return self.cleaned_data_df


if __name__ == '__main__':
    spider = FinvizSpider()
    spider.scrape_data(max_pages='all')   # put 'all' or an integer: e.g. 5
    data = spider.clean_data()
    time_stamp = pd.datetime.now().strftime('%Y%m%d_%H%m')
    data.to_csv(f'Finviz_data_{time_stamp}.csv')
