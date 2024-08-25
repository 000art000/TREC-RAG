import sys
import os

from pyserini.search.lucene import LuceneSearcher


ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

from src.models.monoT5 import MonoT5


class Retriever:
    def __init__(self, corpus, index):
        self.docs_ids = []
        self.searcher = LuceneSearcher.from_prebuilt_index(corpus)
        self.ranker = MonoT5(index , device="cuda")
        self.index = index

    def search(self, query, k=100):
        docs = self.searcher.search(query, k=100)
        retrieved_docid = [i.docid for i in docs]
        docs_text = [
            eval(self.searcher.doc(docid).raw())
            for _, docid in enumerate(retrieved_docid)
        ]
        ranked_doc = self.ranker.rerank(query, docs_text)[:k]
        docids = [i["docid"] for i in ranked_doc]
        scores = [i["score"] for i in ranked_doc]
        docs = [
            eval(self.searcher.doc(docid).raw()) for _, docid in enumerate(docids) # on . raw()
        ]
        #text contient docs_text qui est le doc complet url ...
        """        docs = [
            {"id": docids[i], "text": docs_text[i], "score": scores[i]}
            for i in range(len(docids))
        ]"""

        result = []

        for i in range(len(docids)) : 

            if self.index == "segmented" :
                result.append(
                    {
                        "docid" : docs[i]['docid'] ,
                        "score" : scores[i] ,
                        'docs' : {
                            "url" : docs[i]['url'],
                            "title" : docs[i]['title'],
                            "headings" : docs[i]['headings'],
                            "segment" : docs[i]['segment'],
                            "start_char" : docs[i]['start_char'], 
                            "end_char" : docs[i]['end_char']
                        }
                    }
                )
            elif self.index == "complete" :
                result.append(
                    {
                        "docid" : docs[i]['docid'] ,
                        "score" : scores[i] ,
                        'docs' : {
                            "url" : docs[i]['url'],
                            "title" : docs[i]['title'],
                            "headings" : docs[i]['headings'],
                            "body" : docs[i]['body'],
                        }
                    }
                )

            """docs_text[i] ['score'] = scores[i]
            docs_text[i]['id'] = docs_text[i]['docid'] 
            del docs_text[i]['docid']
            result.append(docs_text[i])"""

        return result

    def process(self, query, **kwargs):
        docs_text = self.search(query, **kwargs)
        return f"\n[DOCS] {docs_text} [/DOCS]\n"
