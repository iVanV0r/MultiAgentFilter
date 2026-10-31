"""Читает ВСЕ страницы PDF и сохраняет их номера при разделении текста по разделам.
Не ограничивается первыми страницами; список литературы можно исключить только из анализа.
"""
from __future__ import annotations
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
import fitz

HEADINGS = [
    ('abstract',r'^(?:abstract|аннотация|резюме)\s*[:.]?$'),
    ('introduction',r'^(?:(?:[IVX]+|\d+(?:\.\d+)*)[.\s]+)?(?:introduction|введение)\s*[:.]?$'),
    ('related_work',r'^(?:(?:[IVX]+|\d+(?:\.\d+)*)[.\s]+)?(?:related\s+work|background|literature\s+review|обзор\s+литературы|анализ\s+литературы)\s*[:.]?$'),
    ('methods',r'^(?:(?:[IVX]+|\d+(?:\.\d+)*)[.\s]+)?(?:methods?(?:\s+and\s+materials)?|methodology|approach|material[s]?\s+and\s+methods|материалы\s+и\s+методы|методика|методы\s+исследования)\s*[:.]?$'),
    ('results',r'^(?:(?:[IVX]+|\d+(?:\.\d+)*)[.\s]+)?(?:results?(?:\s+and\s+discussion)?|experiments?|experimental\s+results|результаты(?:\s+и\s+обсуждение)?|эксперименты)\s*[:.]?$'),
    ('discussion',r'^(?:(?:[IVX]+|\d+(?:\.\d+)*)[.\s]+)?(?:discussion|обсуждение)\s*[:.]?$'),
    ('limitations',r'^(?:(?:[IVX]+|\d+(?:\.\d+)*)[.\s]+)?(?:limitations(?:\s+and\s+future\s+work)?|threats\s+to\s+validity|ограничения(?:\s+исследования)?|перспективы)\s*[:.]?$'),
    ('conclusion',r'^(?:(?:[IVX]+|\d+(?:\.\d+)*)[.\s]+)?(?:conclusions?(?:\s+and\s+future\s+work)?|concluding\s+remarks|заключение|выводы)\s*[:.]?$'),
    ('references',r'^(?:(?:[IVX]+|\d+(?:\.\d+)*)[.\s]+)?(?:references|bibliography|список\s+литературы|литература|источники)\s*[:.]?$'),
    ('appendix',r'^(?:appendi(?:x|ces)(?:\s+[A-Z\d]+)?|приложение(?:\s+\d+)?)\s*[:.]?$'),
]

def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text.replace('\xad','').replace('\u200b',''))
    # Убираем дефисы переноса строки только между буквами; обычные дефисы сохраняем.
    text = re.sub(r'(?<=[A-Za-zА-Яа-яЁё])-\s*\n\s*(?=[a-zа-яё])','',text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fix_mojibake(txt: str) -> tuple[str, bool]:
    """Исправляет русский текст, ошибочно извлечённый из cp1251 как Latin-1.
    Корректные русскоязычные и англоязычные страницы не изменяет.
    """
    ru = len(re.findall('[А-Яа-яЁё]', txt))
    bad = len(re.findall('[À-ÿ]', txt))
    if bad < 40 or ru > bad/3: return txt,False
    try:
        fixed = txt.encode('latin-1').decode('cp1251')
    except (UnicodeError,ValueError):
        # Смешанные управляющие символы могут мешать декодированию; заменяем только фрагменты.
        def conv(m):
            try: return m.group(0).encode('latin-1').decode('cp1251')
            except UnicodeError: return m.group(0)
        fixed = re.sub(r'[À-ÿ]{3,}',conv,txt)
    if len(re.findall('[А-Яа-яЁё]',fixed)) > ru+bad*0.3: return fixed,True
    return txt,False

def _blocks(page):
    blocks=[]
    for b in page.get_text('blocks'):
        if len(b)<7 or b[6]!=0:continue
        t=b[4].strip()
        if t: blocks.append((b[0],b[1],b[2],b[3],t))
    middle=page.rect.width/2
    narrow=[x for x in blocks if x[2]-x[0] < page.rect.width*0.68]
    left=sum(1 for x in narrow if (x[0]+x[2])/2<middle)
    right=sum(1 for x in narrow if (x[0]+x[2])/2>=middle)
    if min(left,right)>=3:
        # Заголовки на всю ширину страницы и последовательные блоки двухколоночного текста.
        blocks.sort(key=lambda b:(0 if b[2]-b[0] > page.rect.width*.73 and b[1]<page.rect.height*.25 else 1,
                                  0 if (b[0]+b[2])/2 < middle else 1,b[1],b[0]))
    else:blocks.sort(key=lambda b:(b[1],b[0]))
    return [x[4] for x in blocks]

def _kind(line):
    s=normalize(line).strip(' :.-')
    if len(s)>105 or len(s)<4:return None
    for kind,regex in HEADINGS:
        if re.fullmatch(regex,s,flags=re.I):return kind
    return None

@dataclass
class Chunk:
    chunk_id: str
    section: str
    pages: list[int]
    text: str


def extract_pdf(path: Path, chunk_chars=8500, overlap_chars=350, include_references=False):
    path=Path(path)
    doc=fitz.open(path)
    pages=[]; sections=[]; current='front_matter'; current_lines=[]; page_ids=[]; fixed_pages=[]
    for ix,page in enumerate(doc,1):
        txt='\n'.join(_blocks(page))
        txt,was_fixed=fix_mojibake(txt)
        fixed_pages.append(ix if was_fixed else None)
        pages.append({'page':ix,'text':normalize(txt),'characters':len(normalize(txt)),
                      'suspected_bad_extraction':len(normalize(txt))<75,'mojibake_fixed':was_fixed})
        for block in txt.splitlines():
            heading=_kind(block)
            if heading:
                if current_lines:
                    sections.append({'section':current,'pages':sorted(set(page_ids)), 'text':normalize('\n'.join(current_lines))})
                    current_lines=[];page_ids=[]
                current=heading
                continue
            current_lines.append(block)
            page_ids.append(ix)
        # Сохраняем границы страниц для фрагментов текста, переходящих на другую страницу.
        if current_lines:
            # Храним текст отдельно для каждой страницы, чтобы точно проверять цитаты.
            sections.append({'section':current,'pages':sorted(set(page_ids)), 'text':normalize('\n'.join(current_lines))})
            current_lines=[];page_ids=[]
    if current_lines:sections.append({'section':current,'pages':sorted(set(page_ids)),'text':normalize('\n'.join(current_lines))})
    if not include_references:
        sections=[s for s in sections if s['section']!='references']
    chunks=[]; idx=0
    for section in sections:
        text=section['text']; pags=section['pages']
        if not text:continue
        off=0
        while off<len(text):
            end=min(len(text),off+chunk_chars)
            if end<len(text):
                br=text.rfind('. ',max(off+chunk_chars//2,end-300),end)
                if br!=-1:end=br+1
            part=text[off:end]
            chunks.append(Chunk(f'C{idx:04}',section['section'],pags,f'[SOURCE PAGE {pags[0]} | SECTION {section["section"]}]\n{part}'))
            idx+=1
            if end==len(text):break
            off=max(end-overlap_chars,off+1)
    all_text=' '.join(x['text'] for x in pages)
    count_pages=len(doc)
    doc.close()
    return {'paper_id':path.stem,'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'parse_config':{'chunk_chars':chunk_chars,'overlap_chars':overlap_chars,'include_references':include_references},
            'pages':pages,'sections':sections,'chunks':[asdict(x) for x in chunks],
            'pdf_pages':count_pages,'processed_pages':len(pages),
            'total_characters':len(all_text),'fixed_mojibake_pages':[x for x in fixed_pages if x],
            'language':('ru' if len(re.findall('[А-Яа-яЁё]',all_text))>len(re.findall('[A-Za-z]',all_text))*.30 else 'en')}

def _valid_quote(quote,txt):
    a=normalize(quote).casefold(); b=normalize(txt).casefold()
    if len(a)<18:return False
    if a in b:return True
    # Учитываем различия в оформлении и переносах PDF, не допуская вымышленных цитат.
    canonical=lambda s:re.sub(r'[^\w]+','',s,flags=re.UNICODE)
    return canonical(a) in canonical(b)

def verify_evidence(problems,pages):
    verified=[]
    page_map={p['page']:p['text'] for p in pages}
    for problem in problems if isinstance(problems,list) else []:
        if not isinstance(problem,dict):continue
        p=dict(problem); evidence=[]
        items=p.get('evidence',[])
        for ev in items if isinstance(items,list) else []:
            if not isinstance(ev,dict):continue
            try:pg=int(ev.get('page'))
            except (ValueError,TypeError):pg=0
            item={'page':pg,'quote':str(ev.get('quote','')),'verified':_valid_quote(str(ev.get('quote','')),page_map.get(pg,''))}
            evidence.append(item)
        p['evidence']=evidence
        p['grounded']=bool(evidence) and any(x['verified'] for x in evidence)
        verified.append(p)
    return verified

def write_json(path,data):
    """Атомарно сохраняет JSON: при прерывании работы в Windows не остаётся частично записанного файла."""
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def main():
    import argparse
    import logging
    import time
    from settings import INPUT_PDFS, OUTPUT, OPTIONS
    parser = argparse.ArgumentParser(description='Разделение всех PDF по страницам и разделам')
    parser.add_argument('--limit', type=int, default=0, help='Обработать только первые N PDF')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    folder = OUTPUT / 'processed'
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_PDFS.glob('*.pdf'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'PDF не найдены в {INPUT_PDFS}')
    summary = []
    t0 = time.perf_counter()
    for i, path in enumerate(files, start=1):
        logging.info('[%s/%s] Чтение PDF: %s', i, len(files), path.name)
        try:
            start = time.perf_counter()
            doc = extract_pdf(path, OPTIONS['chunk_chars'], OPTIONS['overlap_chars'], OPTIONS['include_references'])
            doc['parse_seconds'] = time.perf_counter() - start
            write_json(folder / (path.stem+'.json'), doc)
            summary.append({'paper_id':doc['paper_id'], 'pages':doc['pdf_pages'], 'chunks':len(doc['chunks']), 'seconds':round(doc['parse_seconds'],2)})
            logging.info('  Готово: страниц=%d, фрагментов=%d, время=%.1f с', doc['pdf_pages'], len(doc['chunks']), doc['parse_seconds'])
        except Exception as exc:
            logging.exception('Не удалось обработать %s: %s', path, exc)
            summary.append({'paper_id':path.stem, 'error':str(exc)})
    write_json(folder / 'manifest.json', summary)
    logging.info('Итого PDF: %s, обработка %.1f с, выход: %s', len(files), time.perf_counter()-t0, folder)
    if any('error' in x for x in summary):
        raise SystemExit('Некоторые PDF не обработаны; см. сообщения выше.')


if __name__ == '__main__':
    main()
