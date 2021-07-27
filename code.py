import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# tf_model = TfidfVectorizer().fit(docu)
# sparse_result = tf_model.transform(docu)     # 得到tf-idf矩阵，稀疏矩阵表示法
# # print(sparse_result)
# print(tf_model.vocabulary_)
# # result = sparse_result.todense()
# # print(result[0])
# res = []
# for i in docu:
#     i = jieba.cut(i)
#     print(list(i))
#     break

import re
def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       # 去除网址
    text = text.replace("转发微博", "")       # 去除无意义的词语
    text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格
    text = text.replace('，','')
    return text.strip()

def tf_model():
    docu = ['@小艳子kiki @光影魔术师之择日而栖 @就是爱黑巧克力 尝试新的外景风格，亲们，我有木有拍婚纱照的潜质',
            '大闸蟹amp;amp;红宝石 幸福呀！',
            '2011.11.11预告片：光棍节的可爱小新娘，时不时流露的羞涩和窃喜......1.'
            ]
    docu = [' '.join(jieba.cut(clean(i))) for i in docu]
    print(docu)
    tfidf_model = TfidfVectorizer().fit(docu)
    sparse_result = tfidf_model.transform(docu)  # 得到tf-idf矩阵，稀疏矩阵表示法
    print(sparse_result.todense()[0])


if __name__ == '__main__':
    # a = clean('@小艳子kiki @光影魔术师之择日而栖 @就是爱黑巧克力 尝试新的外景风格，亲们，我有木有拍婚纱照的潜质')
    # a = jieba.cut(a)
    # a = ' '.join(a)
    # print(a)
    # a = str(list(jieba.cut(a)))

    # print(a)
    # tf_model()
    def merge_sort(alist):
        if len(alist) <= 1:
            return alist
        # 二分分解
        num = len(alist)/2
        left = merge_sort(alist[:num])
        right = merge_sort(alist[num:])
        # 合并
        return merge(left,right)

    def merge(left, right):
        '''合并操作，将两个有序数组left[]和right[]合并成一个大的有序数组'''
        #left与right的下标指针
        l, r = 0, 0
        result = []
        while l<len(left) and r<len(right):
            if left[l] < right[r]:
                result.append(left[l])
                l += 1
            else:
                result.append(right[r])
                r += 1
        result += left[l:]
        result += right[r:]
        return result

    alist = [54,26,93,17,77,31,44,55,20]
    sorted_alist = merge_sort(alist)
    print(sorted_alist)
