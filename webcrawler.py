import wikipedia
import codecs

file_count = 0


def save_to_file(url, text):
    global file_count
    with codecs.open('data/file' + str(file_count), 'w', 'utf-8') as f:
        f.write(text)
    with open('data/data', 'a') as d:
        d.write('file' + str(file_count) + " " + url + '\n')
    file_count += 1


for i in range(1, 2501):
    query = wikipedia.random(1)
    try:
        article = wikipedia.page(query)
    except:
        continue
    save_to_file(article.url, article.summary)
