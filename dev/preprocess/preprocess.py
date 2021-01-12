import spacy


class DependencyG:
    def __init__(self, text):

        self.text_parsed = self.__dep_parse(text)
        # convert dependency output to graph
        self.u, self.v, self.dep, self.r, self.sent_end = self.__dep_graph()

    def __dep_parse(self, text):
        # parse using Spacy.
        # TODO: give the ability to parse using external function.
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return doc

    def __dep_graph(self):
        # Construct the graph from the parser
        # graph: G((u1,v1), dep). u1 --dep--> v1
        u1 = []
        v1 = []
        dep = []
        is_root = []  # 1 if u is the root
        is_sent_end = []  # 1 if u is the end of the sentence

        # fill u1, v1, dep, is_root, is_sent_end
        for token in self.text_parsed:
            if token.dep_ == "ROOT":
                is_root.append(1)
                is_sent_end.append(0)
                continue

            if token.is_sent_end:
                is_sent_end.append(1)
            else:
                is_sent_end.append(0)

            # leaves-root (word--dep-->head)
            is_root.append(0)
            u1.append(token.i)
            v1.append(token.head.i)
            dep.append(token.dep)

        return u1, v1, dep, is_root, is_sent_end
