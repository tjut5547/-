import jieba

class Data(object):
    def __init__(self, normal, spam):
        normal_file = open(normal, "r", encoding='utf-8')
        spam_file = open(spam, "r", encoding='utf-8')

        self.data = []
        self.label = []
        for line in normal_file:
            _, content = line.split('::')
            content = ' '.join(list(jieba.cut(content)))
            self.data.append(content)
            self.label.append(1)

        for line in spam_file:
            _, content = line.split('::')
            # content = ' '.join(list(jieba.cut(content)))
            content = ' '.join(list(jieba.cut(content)))
            self.data.append(content)
            self.label.append(0)
