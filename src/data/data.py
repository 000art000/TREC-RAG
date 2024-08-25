import re
from pyserini.search.lucene import LuceneSearcher
import csv

def remove_special_spaces(text):
    """# This regex finds spaces that are not followed by a letter
    pattern = r'[\t\n\r\f\v\s](\W+|$)'
    # Replace these spaces with an empty string
    result = re.sub(pattern, ' ', f'{text}')
    result = result.replace('\s',' ')
    result = result.replace('\n',' ')
    result = result.replace('\r',' ')
    result = result.replace('\t',' ')
    result = result.replace('\v',' ')
    result = result.replace('\f',' ')"""
    text = re.sub(r'\W', ' ', text)
    # Supprimer les caractères qui commencent par une barre oblique inversée
    text = re.sub(r'\\.', '', text).strip()
    return text

def get_passage(query_id, path_passage, CONFIG):

    """for filename in os.listdir(path_passage):
        if filename.endswith('.tsv'):
            if re.search(r'_(\d+)\.tsv$', filename).group(1) == query_id :
                passage_file = path_passage + '/' + filename"""

    b = False
    with open(path_passage, mode='r', encoding='utf-8') as fichier:
        lecteur_tsv = csv.reader(fichier, delimiter='\t')
        result = []
        for ligne in lecteur_tsv:
            if ligne[CONFIG['passage_column']['topic_id']] == query_id: 
                result.append(ligne)
                b = True
            else :
                if b :
                    return result
        return result

def extract_sentences_and_citations(text):
    # Définir le motif pour capturer les phrases et les listes de citations
    pattern = re.compile(r'(.*?)\s*(\[\d+(?:,\s*\d+)*\])\.')
    # Trouver toutes les correspondances dans le texte
    matches = pattern.findall(text)
    # Extraire les phrases et les listes de citations
    results = []
    for match in matches:
        sentence = match[0].strip()
        citation_list = match[1].strip()
        results.append({"text": sentence, "citations": citation_list})
    
    return results

def format_support_passages(context_list, searcher,CONFIG):

    """passages = []
    for i in range(len(passages_text)):
        passages.append(
            {"idx": i + 1, "docid": passages_text[i]['id'], "text": passages_text[i]['body']}
        )
    return passages"""

    context_text = []
    pos_docid = CONFIG['passage_column']['docid']

    for i in range(len(context_list)) :
        doc = eval(searcher.doc(context_list[i][pos_docid]).raw())
        context_text.append(doc['docid'])

    return context_text

def remove_all_newlines(text):
    # Remplace tous les retours à la ligne par un espace
    text = text.replace('\n', ' ')
    
    # Optionnel: supprimer les espaces en trop au début et à la fin du texte
    text = text.strip()
    
    return text

def prepare_contexts(context_list, CONFIG, corpus, searcher, citation=True):

    context_text = []
    pos_docid = CONFIG['passage_column']['docid']

    if corpus == "segmented" : passage_key = 'segment'
    elif corpus == "complete" : passage_key = 'body' 

    if citation:
        for i in range(len(context_list)) :
            doc = eval(searcher.doc(context_list[i][pos_docid]).raw())
            doc = "[" + str((i + 1)) + "] " + remove_all_newlines(doc[passage_key])
            context_text.append(doc)
    else:
        for i in range(len(context_list)) :
            doc = eval(searcher.doc(context_list[i][pos_docid]).raw())
            doc = remove_all_newlines(doc[passage_key])
            context_text.append(doc)
        
    return "\n".join(context_text)