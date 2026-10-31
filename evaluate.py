"""Precision/Recall/F1 относительно человеческого эталона, плюс ручная проверка пар."""
import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from prepare import write_json
from settings import EXCLUDED_PAPER_IDS, HUMAN_GOLD, OUTPUT, SIMILARITY_THRESHOLD, EMBEDDING_MODEL, EVALUATION_MATCHER


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path,rows,columns=None):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and columns is None:
        return
    keys=columns or list(rows[0])
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def aggregate(rows):
    if not rows:
        return None
    tp=sum(x['tp'] for x in rows)
    fp=sum(x['fp'] for x in rows)
    fn=sum(x['fn'] for x in rows)
    valid=[x['f1'] for x in rows if x['f1'] is not None]
    return {'articles_x_repetitions':len(rows), 'tp':tp, 'fp':fp, 'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'macro_f1':float(np.mean(valid)) if valid else None}


def semantic_embeddings(texts, model=EMBEDDING_MODEL):
    """Локальные многоязычные embeddings через Ollama /api/embed.

    Не используются в извлечении — только после эксперимента для оценки.
    """
    import requests
    from settings import OLLAMA_URL
    vectors=[]
    for k in range(0,len(texts),16):
        batch=texts[k:k+16]
        response=requests.post(OLLAMA_URL.rstrip('/')+'/api/embed',json={
            'model':model,'input':batch,'truncate':False,'keep_alive':'2m'
        },timeout=300)
        if response.status_code >=400:
            raise RuntimeError(f'Ошибка Ollama embeddings HTTP {response.status_code}: {response.text[:350]}; '
                               f'установите: ollama pull {model}')
        part=response.json().get('embeddings',[])
        if len(part)!=len(batch):
            raise RuntimeError('Ollama /api/embed вернула неполный набор векторов')
        vectors.extend(part)
    return np.asarray(vectors,dtype=float)


def problem_similarity(pred,gold,matcher='semantic',embedding_model=EMBEDDING_MODEL):
    if not pred or not gold:return np.zeros((len(pred),len(gold)))
    texts=[p['statement'] for p in pred]+[p['statement'] for p in gold]
    if matcher=='semantic':
        vectors=semantic_embeddings(texts,embedding_model)
        return cosine_similarity(vectors[:len(pred)],vectors[len(pred):])
    if matcher!='lexical':raise ValueError(f'Неизвестный matcher: {matcher}')
    vect=TfidfVectorizer(analyzer='char',ngram_range=(2,5),lowercase=True)
    array=vect.fit_transform(texts)
    return cosine_similarity(array[:len(pred)],array[len(pred):])

def calculate(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else (None if tp+fn==0 else 0.0),
            'recall':tp/(tp+fn) if tp+fn else (None if tp+fp==0 else 0.0),
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def match_auto(pred,gold,threshold,matcher=EVALUATION_MATCHER,embedding_model=EMBEDDING_MODEL):
    sim=problem_similarity(pred,gold,matcher,embedding_model)
    links=[]
    if sim.size:
        # +100 требует сначала назначать пары выше порога, а затем максимизировать сходство.
        a,b=linear_sum_assignment(-((sim>=threshold).astype(float)*100+sim))
        for i,j in zip(a,b):
            if sim[i,j]>=threshold:
                links.append((int(i),int(j),float(sim[i,j])))
    scores=calculate(len(links),len(pred)-len(links),len(gold)-len(links))
    return scores,links,sim


def read_runs(folder):
    rows=[]
    for mode in ('single','multi'):
        for path in sorted((folder/mode).glob('*.json')):
            d=read_json(path)
            if d.get('paper_id') not in EXCLUDED_PAPER_IDS:
                rows.append((mode,d))
    return rows


def report_by_group(rows, source):
    result={'basis':source,'by_mode':{},'by_language':{}}
    for mode in ('single','multi'):
        subset=[r for r in rows if r['mode']==mode]
        result['by_mode'][mode]=aggregate(subset)
        result['by_language'][mode]={lang:aggregate([r for r in subset if r['language']==lang]) for lang in ('ru','en')}
    # Paired difference per unique article, aggregating repeated runs BEFORE uncertainty.
    paired=defaultdict(dict)
    for row in rows:
        paired[(row['paper_id'],row['repetition'])][row['mode']]=row
    per_paper=defaultdict(list)
    paired_data=[]
    for (pid,rep), modes in sorted(paired.items()):
        if 'single' in modes and 'multi' in modes:
            a,b=modes['single']['f1'],modes['multi']['f1']
            if a is not None and b is not None:
                d=float(b-a)
                per_paper[pid].append(d)
                paired_data.append({'paper_id':pid,'repetition':rep,'single_f1':a,'multi_f1':b,'delta_f1':d})
    if per_paper:
        means=np.asarray([float(np.mean(values)) for values in per_paper.values()])
        rng=np.random.default_rng(42)
        boots=[float(rng.choice(means,size=len(means),replace=True).mean()) for _ in range(2000)]
        result['paired_difference']={
            'n_independent_articles':len(means),
            'mean_multi_minus_single_f1':float(means.mean()),
            'bootstrap_95_ci':[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
        }
        try:
            from scipy.stats import wilcoxon
            result['paired_difference']['wilcoxon_p_exploratory']=float(wilcoxon(means).pvalue) if any(means) else 1.0
        except ValueError:
            result['paired_difference']['wilcoxon_p_exploratory']=None
    result['paired_runs']=paired_data
    return result


def auto_evaluate(folder, gold, threshold, matcher=EVALUATION_MATCHER, embedding_model=EMBEDDING_MODEL):
    scores=[];matches=[];review=[]
    source=read_runs(folder)
    if not source:
        raise SystemExit('Не найдены результаты моделей. Сначала запустите python run.py.')
    for mode, result in source:
        pid=result['paper_id'];rep=result.get('repetition',1)
        if pid not in gold:
            logging.warning('Нет эталона для %s',pid)
            continue
        g=gold[pid]['problems'];p=[v for v in result.get('problems',[]) if v.get('statement')]
        row, links, matrix = match_auto(p,g,threshold,matcher,embedding_model)
        # Диагностическое сходство сохраняется даже при отсутствии пар выше порога.
        # Это не F1 и не ручная оценка TP.
        pred_similarity = float(np.mean(np.max(matrix, axis=1))) if len(p) and len(g) else None
        gold_similarity = float(np.mean(np.max(matrix, axis=0))) if len(p) and len(g) else None
        matched_similarity = float(np.mean([similarity for _,_,similarity in links])) if links else None
        scores.append({'paper_id':pid,'language':gold[pid]['language'],'mode':mode,'repetition':rep,**row,
                       'mean_prediction_similarity':round(pred_similarity,4) if pred_similarity is not None else '',
                       'mean_gold_similarity':round(gold_similarity,4) if gold_similarity is not None else '',
                       'mean_matched_similarity':round(matched_similarity,4) if matched_similarity is not None else ''})
        by_prediction={i:(j,similarity) for i,j,similarity in links}
        if not p:
            review.append({'paper_id':pid,'language':gold[pid]['language'],'mode':mode,'repetition':rep,
                           'prediction_id':'(none)','statement':'','suggested_gold_id':'','similarity':'',
                           'gold_id':'','reviewed_status':'NONE','comment':'Модель не нашла ни одной проблемы'})
        for i,item in enumerate(p):
            j,score=by_prediction.get(i,(None,None))
            guess=g[j]['id'] if j is not None else ''
            nearest_index=int(np.argmax(matrix[i])) if len(g) else None
            nearest_similarity=float(matrix[i,nearest_index]) if nearest_index is not None else None
            match={'paper_id':pid,'language':gold[pid]['language'],'mode':mode,'repetition':rep,
                   'prediction_id':f'M{i+1}','statement':item['statement'],'suggested_gold_id':guess,
                   'similarity':round(score,4) if score is not None else '',
                   'nearest_gold_id':g[nearest_index]['id'] if nearest_index is not None else '',
                   'nearest_similarity':round(nearest_similarity,4) if nearest_similarity is not None else '',
                   'gold_id':'','reviewed_status':'','comment':''}
            review.append(match)
            matches.append(dict(match,gold_statement=g[j]['statement'] if j is not None else ''))
        logging.info('%-6s | %-33s | повтор %s | TP=%s FP=%s FN=%s F1=%.3f',mode,pid,rep,row['tp'],row['fp'],row['fn'],row['f1'] or 0.0)
    out=folder/'evaluation_auto'
    write_csv(out/'metrics.csv',scores)
    write_csv(out/'matches.csv',matches)
    review_path=folder/'review.csv'
    if not review_path.exists():
        write_csv(review_path,review)
        logging.info('Шаблон ручной проверки: %s',review_path)
    else:
        logging.info('Существующий review.csv сохранён без перезаписи.')
    source_name = 'automatic_multilingual_embedding_similarity_only' if matcher=='semantic' else 'automatic_character_similarity_only'
    summ=report_by_group(scores,source_name)
    summ['matching_method']=matcher
    summ['embedding_model']=embedding_model if matcher=='semantic' else None
    summ['matching_threshold']=threshold
    summ['caution']='Семантические embeddings позволяют сравнивать смысл, но не подтверждают истинность TP. Для научного вывода проверьте пары в review.csv.' if matcher=='semantic' else 'Символьное сходство НЕ является оценкой смысла. Используйте --matcher semantic и ручной review.csv.'
    write_json(out/'summary.json',summ)
    logging.info('Автоматические показатели: Single F1=%s | Multi F1=%s',
                 (summ['by_mode']['single'] or {}).get('micro_f1'),(summ['by_mode']['multi'] or {}).get('micro_f1'))
    return scores


def reviewed_evaluate(folder,gold,review_file):
    with Path(review_file).open('r',encoding='utf-8-sig',newline='') as f:
        reviews=list(csv.DictReader(f))
    if not reviews:
        raise SystemExit('Файл ручной проверки пустой')
    allowed={'TP','FP','Partial','Duplicate','NONE'}
    groups=defaultdict(list)
    expected={}
    for mode, data in read_runs(folder):
        pid=data['paper_id'];rep=int(data.get('repetition',1))
        if pid not in gold:continue
        expected[(pid,mode,rep)]=len(data.get('problems',[]))
    for i,r in enumerate(reviews,2):
        if r.get('paper_id') in EXCLUDED_PAPER_IDS: continue
        status=r['reviewed_status'].strip()
        if status not in allowed:
            raise ValueError(f'Строка {i}: укажите reviewed_status = TP/FP/Partial/Duplicate/NONE')
        pid=r['paper_id'];mode=r['mode'];rep=int(r['repetition'])
        key=(pid,mode,rep)
        if key not in expected:
            raise ValueError(f'Строка {i}: нет такого результата модели: {key}')
        if status in {'TP','Partial','Duplicate'} and r.get('gold_id','') not in {x['id'] for x in gold[pid]['problems']}:
            raise ValueError(f'Строка {i}: gold_id обязателен и должен принадлежать статье')
        if status in {'FP','NONE'} and r.get('gold_id','').strip():
            raise ValueError(f'Строка {i}: gold_id для FP/NONE должен быть пуст')
        if status=='NONE' and r.get('statement','').strip():
            raise ValueError(f'Строка {i}: NONE только для статьи без предсказаний')
        groups[key].append(r)
    if set(groups)!=set(expected):
        raise ValueError('В CSV не хватает строк для некоторых обработанных статей')
    scores=[]
    for (pid,mode,rep), group in sorted(groups.items()):
        n=expected[(pid,mode,rep)]
        if len(group)!=max(1,n):
            raise ValueError(f'{pid}/{mode}/{rep}: ожидалось {max(1,n)} строк, найдено {len(group)}')
        if len({r['prediction_id'] for r in group})!=len(group):
            raise ValueError(f'{pid}/{mode}/{rep}: повторяется prediction_id')
        if n==0 and (group[0]['reviewed_status']!='NONE' or len(group)!=1):
            raise ValueError(f'{pid}/{mode}/{rep}: без предсказаний должен быть один NONE')
        if n>0 and any(r['reviewed_status']=='NONE' for r in group):
            raise ValueError(f'{pid}/{mode}/{rep}: NONE нельзя использовать при наличии предсказания')
        tp_ids=[r['gold_id'] for r in group if r['reviewed_status']=='TP']
        if len(set(tp_ids))!=len(tp_ids):
            raise ValueError(f'{pid}/{mode}/{rep}: один эталон нельзя засчитывать TP дважды')
        tp=len(tp_ids);fp=n-tp;fn=len(gold[pid]['problems'])-tp
        partial=sum(r['reviewed_status']=='Partial' for r in group)
        dupl=sum(r['reviewed_status']=='Duplicate' for r in group)
        scores.append({'paper_id':pid,'language':gold[pid]['language'],'mode':mode,'repetition':rep,
                       **calculate(tp,fp,fn),'partial':partial,'duplicate':dupl})
    out=folder/'evaluation_reviewed'
    write_csv(out/'metrics.csv',scores)
    summary=report_by_group(scores,'human_reviewed_strict_TP')
    summary['partial_policy']='Partial и Duplicate не засчитываются как TP (FP со стороны ответа и FN для ненайденного эталона).'
    write_json(out/'summary.json',summary)
    logging.info('Строгая оценка по проверенным соответствиям готова: %s',out)


def main():
    parser=argparse.ArgumentParser(description='Оценка извлечённых проблем по 96 человеческим эталонам')
    parser.add_argument('--name',default='full',help='Имя серии из run.py')
    parser.add_argument('--threshold',type=float,default=SIMILARITY_THRESHOLD,help='Порог семантического совпадения, подбирайте на development выборке')
    parser.add_argument('--matcher',choices=['semantic','lexical'],default=EVALUATION_MATCHER,help='semantic — локальные многоязычные embeddings Ollama; lexical — только диагностический baseline')
    parser.add_argument('--embedding-model',default=EMBEDDING_MODEL,help='Название модели Ollama embeddings')
    parser.add_argument('--review',type=str,default='',help='Путь к вручную заполненному review.csv')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(message)s',datefmt='%H:%M:%S')
    if args.threshold<0 or args.threshold>1:
        parser.error('Порог должен находиться от 0 до 1')
    gold={row['paper_id']:row for row in read_json(HUMAN_GOLD)['papers']}
    folder=OUTPUT/args.name
    if args.review:
        reviewed_evaluate(folder,gold,args.review)
    else:
        auto_evaluate(folder,gold,args.threshold,args.matcher,args.embedding_model)
    logging.info('Следующий этап: python visualize.py --name %s',args.name)


if __name__=='__main__':
    main()
