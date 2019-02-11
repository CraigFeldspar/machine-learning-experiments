import scrapy


class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://www.wanadev.fr/blog/?page=' + str(i) for i in range(0, 14)]

    def parse_content(self, response):
        result = dict()
        result['title'] = response.css('.container-blog-post-info + p::text').extract_first()
        result['text'] = ""
        for element in response.css('.blog-post-content-container *::text').extract():
            result['text'] += element + "\n"
        yield result

    def parse(self, response):
        # for title in response.css('.latest-post-one h6 a'):
        #     yield {'title': title.css('::text').extract_first()}
        print('Processing..' + response.url)

        for article in response.css('.latest-post-one h6 a'):
            yield response.follow(article.css('::attr(href)').extract_first(), self.parse_content)
