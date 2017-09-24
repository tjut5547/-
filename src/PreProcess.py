import jieba

class stopword(object):
    def __init__(self, file):
        fd = open(file, encoding='utf-8')
        self.data = []
        for word in fd:
            self.data.append(word.strip())


class Data(object):
    def __init__(self, normal, spam, stop):
        # 去停用词
        def remove_stopwords(stop_list, data_list):
            result = []
            # 这里面的leave_word不能作为最终的特征，因为leave_word当中每个词只出现一次
            # 当多个非停用词出现在一句话中的时候，信息就会损失
            leave_word = set(data_list) - set(stop_list)
            for word in data_list:
                if word in leave_word:
                    result.append(word)
            return result

        self.data = []
        self.label = []
        normal_file = open(normal, "r", encoding='utf-8')
        spam_file = open(spam, "r", encoding='utf-8')
        self.stopwords = set(stopword(stop).data)

        for line in normal_file:
            _, content = line.split('::')
            content = remove_stopwords(self.stopwords, list(jieba.cut(content)))
            content = ' '.join(content)
            self.data.append(content)
            self.label.append(1)

        for line in spam_file:
            _, content = line.split('::')
            content = remove_stopwords(self.stopwords, list(jieba.cut(content)))
            content = ' '.join(content)
            self.data.append(content)
            self.label.append(0)

