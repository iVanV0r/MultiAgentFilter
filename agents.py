"""Сравнение single Qwen с триадой Qwen/Phi/Gemma; аудит потерь кандидатов."""
from __future__ import annotations
import logging
import re
import time
from prepare import verify_evidence, normalize
import prompts


def unique(items, limit=7):
    seen=set();out=[]
    for x in items:
        if not isinstance(x,dict):continue
        s=str(x.get('statement','')).strip()
        key=re.sub(r'\W+','',s.casefold())
        if len(s)<20 or key in seen:continue
        seen.add(key);out.append(x)
        if len(out)>=limit:break
    return out


def local_context(pages,problems,cap=3500):
    tokens=set(re.findall(r'[\w]{4,}',' '.join(p.get('statement','') for p in problems).casefold()))
    cited={e.get('page') for p in problems for e in p.get('evidence',[]) if isinstance(e,dict)}
    rows=[]
    for p in pages:
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])',p['text']):
            if len(sentence)<45:continue
            score=len(tokens&set(re.findall(r'[\w]{4,}',sentence.casefold())))+(3 if p['page'] in cited else 0)
            if score:rows.append((score,p['page'],sentence[:410]))
    rows.sort(key=lambda t:(-t[0],t[1]))
    out=[];used=set();n=0
    for _,page,s in rows:
        key=normalize(s)[:130]
        txt=f'[SOURCE PAGE {page}] {s}'
        if key in used or n+len(txt)>cap:continue
        out.append(txt);used.add(key);n+=len(txt)
    return '\n'.join(out)


def global_context(pages,cap=3300):
    """Обзор с охватом всего документа, а не только начальных страниц."""
    cue=('however','limitation','lack ','underexplored','we address','challenge','gap ',
         'проблем','недостат','огранич','отсутств','нехват','малоизуч')
    rows=[]
    for p in pages:
        parts=re.split(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])',p['text'])
        found=[s for s in parts if 45<=len(s) and any(t in s.casefold() for t in cue)]
        if found:rows.append((p['page'],found[0][:350]))
    if not rows:rows=[(p['page'],p['text'][:300]) for p in pages if p['text']]
    step=max(1,(len(rows)+19)//20)
    out=[];length=0
    for page,s in rows[::step]:
        t=f'[SOURCE PAGE {page}] {s}'
        if length+len(t)>cap:break
        out.append(t);length+=len(t)
    return '\n'.join(out)


def ask(client,models,role,prompt,request_id):
    family='proposer' if role.startswith('proposer') else role
    return client.chat(models[family],role,prompt,request_id=request_id,system=prompts.SYSTEM[family])


def initial_proposals(doc,client,models,cfg,tag):
    rows=[];events=[]
    for i,ch in enumerate(doc['chunks'],1):
        logging.info('  Qwen фрагмент %d/%d | раздел %s | страницы %s',i,len(doc['chunks']),ch['section'],ch['pages'])
        response,perf=ask(client,models,'proposer_chunk',
                          prompts.proposer_chunk(ch['text'],cfg.get('max_candidates_per_chunk',3)),
                          f'{tag}:{ch["chunk_id"]}')
        checked=verify_evidence(response.get('problems',[]),doc['pages'])
        rows.extend(x for x in checked if x['grounded'])
        events.append({'chunk':ch['chunk_id'],'section':ch['section'],'pages':ch['pages'],
                       'generated':len(checked),'grounded':sum(x['grounded'] for x in checked),
                       'raw':checked,'metrics':perf})
        logging.info('  Qwen: кандидатов %d, подтверждено %d',len(checked),events[-1]['grounded'])
    return rows,events


def consolidate(doc,client,models,candidates,cfg,tag):
    candidates=unique(candidates,500)
    if not candidates:return [],[]
    limit=cfg.get('max_problems_per_paper',7);perf=[];level=0
    while len(candidates)>limit and level<7:
        logging.info('  Консолидация %d кандидатов (итерация %d)',len(candidates),level+1)
        batches=[candidates[i:i+5] for i in range(0,len(candidates),5)]
        merged=[]
        for i,batch in enumerate(batches):
            try:
                answer,met=ask(client,models,'proposer_merge',prompts.proposer_merge(batch),
                               f'{tag}:merge:{level}:{i}')
                perf.append(met)
                verified=verify_evidence(answer.get('problems',[]),doc['pages'])
                good=[x for x in verified if x['grounded']]
                if not good:
                    logging.warning('  Merge пуст: не теряю входные проблемы')
                    good=batch
                merged.extend(good)
            except Exception as exc:
                logging.warning('  Merge ошибка: %s; сохраняю входные кандидаты',exc)
                merged.extend(batch)
        new=unique(merged,500)
        if len(new)>=len(candidates):break  # Не заставляем модель сокращать без аргументов.
        candidates=new;level+=1
    if len(candidates)<=limit:return candidates,perf
    # Single-Agent имеет фиксированное максимальное число разных проблем.
    response,met=ask(client,models,'proposer_final',prompts.proposer_final(candidates,limit),f'{tag}:final')
    perf.append(met)
    final=unique([p for p in verify_evidence(response.get('problems',[]),doc['pages']) if p['grounded']],limit)
    if not final:
        logging.warning('  Финальная консолидация пуста, сохраняю исходные подтверждённые')
        final=unique(candidates,limit)
    return final,perf


def rubric_score(verdict):
    rub=verdict.get('rubric',{})
    limits={'gap':3,'relevance':2,'evidence':3,'precision':2}
    if isinstance(rub,dict) and all(k in rub for k in limits):
        try:return sum(max(0,min(hi,int(rub[k]))) for k,hi in limits.items()),'rubric_sum'
        except (ValueError,TypeError):pass
    try:return max(0,min(10,int(verdict.get('score',0)))),'raw_fallback'
    except (ValueError,TypeError):return 0,'missing'


def validated_decisions(candidates,verdict):
    ids={p['candidate_id'] for p in candidates};found={}
    for row in verdict.get('decisions',[]) if isinstance(verdict.get('decisions'),list) else []:
        if not isinstance(row,dict):continue
        i=row.get('candidate_id')
        if i in ids and type(row.get('accept')) is bool and i not in found:found[i]=row
    return found,ids-set(found)


def judge_result(candidates,verdict):
    found,missing=validated_decisions(candidates,verdict)
    approved=[];decisions=[]
    for p in candidates:
        i=p['candidate_id'];row=found.get(i)
        accepted=bool(row is not None and row['accept'] and p.get('grounded'))
        status='accepted' if accepted else ('no_judge_decision' if row is None else 'rejected')
        decisions.append({'candidate_id':i,'statement':p['statement'],'status':status,
                          'reason':str(row.get('reason','')) if row else 'Судья пропустил ID',
                          'grounded':p.get('grounded',False),'evidence':p.get('evidence',[])})
        if accepted:approved.append({k:v for k,v in p.items() if k!='candidate_id'})
    return approved,decisions,missing


def apply_defense(previous,answer,pages,cap=7):
    # Нельзя молча удалять элементы: отсутствие ответа = keep.
    actions={}
    for a in answer.get('actions',[]) if isinstance(answer.get('actions'),list) else []:
        if isinstance(a,dict) and a.get('candidate_id') not in actions:
            actions[a.get('candidate_id')]=a
    out=[];audit=[]
    for original in previous:
        cid=original['candidate_id'];a=actions.get(cid)
        operation=str(a.get('action','')).lower() if a else 'keep'
        reason=str(a.get('reason','')).strip() if a else 'Пропущено действие — сохранено'
        if operation=='remove' and reason:
            audit.append({'candidate_id':cid,'action':'remove','reason':reason,
                          'old_statement':original['statement']})
            continue
        item=dict(original)
        if operation=='rewrite' and len(str(a.get('statement','')).strip())>19:
            item['statement']=a['statement'].strip()
        else:operation='keep'
        audit.append({'candidate_id':cid,'action':operation,'reason':reason,
                      'old_statement':original['statement'],'new_statement':item['statement']})
        out.append(item)
    additions=answer.get('additions',[])
    if not isinstance(additions,list):additions=[]
    added=verify_evidence(additions[:2],pages)
    # Диагностика: неподтверждённые предложения Qwen-защитника сохраняются
    # отдельно от неподтверждённых дополнений Phi.
    for p in added:
        if not p.get('grounded'):
            audit.append({'action':'invalid_evidence_addition',
                          'candidate_id':'', 'statement':p.get('statement',''),
                          'evidence':p.get('evidence',[]),
                          'reason':'Цитата дополнения Qwen не прошла автоматическую проверку'})
    keys={re.sub(r'\W+','',p['statement'].lower()) for p in out}
    next_id=1
    for p in added:
        key=re.sub(r'\W+','',p.get('statement','').lower())
        if not p.get('grounded') or len(p.get('statement',''))<20 or key in keys or len(out)>=cap:continue
        p['candidate_id']=f'N{next_id}'
        while p['candidate_id'] in {x['candidate_id'] for x in out}:
            next_id+=1;p['candidate_id']=f'N{next_id}'
        keys.add(key);out.append(p)
        audit.append({'candidate_id':p['candidate_id'],'action':'addition','reason':'Цитата подтверждена'})
    return out,audit


def debate(doc,client,models,cfg,initial,tag):
    current=[dict(p,candidate_id=f'M{i}') for i,p in enumerate(initial,1)]
    history=[];best=None;best_score=-1;reason='max_rounds'
    for r in range(1,cfg.get('max_rounds',3)+1):
        logging.info('  Раунд %s/%s, кандидатов=%s',r,cfg.get('max_rounds',3),len(current))
        ctx=local_context(doc['pages'],current,cap=cfg.get('debate_context_chars',3500))
        overview=global_context(doc['pages'],cap=cfg.get('global_context_chars',3300))
        critic,cperf=ask(client,models,'critic',prompts.critic(overview,current,[]),f'{tag}:r{r}:critic')
        phi_add=verify_evidence(critic.get('missed_problems',[]),doc['pages'])
        checked_critic=dict(critic,missed_problems=[x for x in phi_add if x.get('grounded')])
        logging.info('  Phi: замечаний=%s, дополнений с цитатами=%s',
                     len(critic.get('objections',[])),len(checked_critic['missed_problems']))
        defense,dperf=ask(client,models,'proposer_defense',
                          prompts.defense(ctx,current,checked_critic,r),f'{tag}:r{r}:defense')
        updated,audit=apply_defense(current,defense,doc['pages'],cfg.get('max_problems_per_paper',7))
        # Проверенные дополнения критика поступают судье, даже если защитник их не перенёс.
        keys={re.sub(r'\W+','',p['statement'].casefold()) for p in updated}
        for p in checked_critic['missed_problems']:
            key=re.sub(r'\W+','',p.get('statement','').casefold())
            if key in keys or len(updated)>=cfg.get('max_problems_per_paper',7):continue
            new=dict(p,candidate_id=f'P{r}_{len(updated)+1}')
            updated.append(new);keys.add(key)
            audit.append({'candidate_id':new['candidate_id'],'action':'critic_addition',
                          'reason':'Новая проблема Phi с подтверждённой цитатой'})
        judge_ctx=local_context(doc['pages'],updated,cap=cfg.get('judge_context_chars',2200))
        verdict,jperf=ask(client,models,'referee',prompts.referee(judge_ctx,updated,critic,cfg.get('stop_score',8)),
                          f'{tag}:r{r}:referee')
        checked,missing=validated_decisions(updated,verdict)
        extra=[]
        if missing:
            logging.warning('  Gemma пропустила %d кандидатов: %s. Запрашиваю решения.',len(missing),sorted(missing))
            needed=[{'candidate_id':p['candidate_id'],'statement':p['statement'],
                     'evidence':p.get('evidence',[])[:1]} for p in updated if p['candidate_id'] in missing]
            try:
                repair,extra_perf=ask(client,models,'referee',prompts.referee_missing(needed,judge_ctx),
                                      f'{tag}:r{r}:referee_repair')
                extra.append(extra_perf)
                old=verdict.get('decisions') if isinstance(verdict.get('decisions'),list) else []
                new=repair.get('decisions') if isinstance(repair.get('decisions'),list) else []
                verdict['decisions']=old+new
            except Exception as exc:logging.warning('  Gemma не дополнила решения: %s',exc)
        accepted,decisions,missing=judge_result(updated,verdict)
        score,source=rubric_score(verdict)
        complete=not missing
        judge_consistency_warning=bool(updated and not accepted and score>=cfg.get('stop_score',8))
        if judge_consistency_warning:
            logging.warning('  Gemma оценила раунд на %s/10, но не приняла ни одной проблемы. Высокая оценка НЕ завершит цикл.',score)
        history.append({'round':r,'critic':critic,'defense':defense,'defense_actions':audit,
                        'referee':verdict,'score':score,'score_source':source,
                        'judge_decisions_complete':complete,'missing_judge_ids':sorted(missing),
                        'judge_consistency_warning':judge_consistency_warning,
                        'candidates':updated,'invalid_evidence_candidates':[p for p in phi_add if not p.get('grounded')],
                        'decisions':decisions,'accepted':accepted,
                        'metrics':[cperf,dperf,jperf]+extra})
        logging.info('  Gemma: score=%s/10, решения=%s/%s, принято=%s',
                     score,len(updated)-len(missing),len(updated),len(accepted))
        if complete and (best is None or score>best_score):
            best_score=score;best={'problems':accepted,'decisions':decisions,'round':r}
        current=updated
        if complete and score>=cfg.get('stop_score',8) and not judge_consistency_warning:
            reason='threshold';break
    if best is None:
        raise RuntimeError('Судья не вернул полный набор решений в пределах раундов; Single сохранён')
    return {'problems':best['problems'],'history':history,'best_score':best_score,
            'best_round':best['round'],'final_decisions':best['decisions'],
            'rounds':len(history),'stopping_reason':reason}


def run_paper(doc,client,models,cfg,tag,*,on_single_ready=None,saved_single=None):
    t0=time.perf_counter()
    if saved_single is None:
        candidates,events=initial_proposals(doc,client,models,cfg,tag)
        proposal,merges=consolidate(doc,client,models,candidates,cfg,tag)
        elapsed=time.perf_counter()-t0
        single={'paper_id':doc['paper_id'],'filename':doc['filename'],'mode':'single','problems':proposal,
                'seconds_to_single':elapsed,'chunk_events':events,'merge_metrics':merges}
        if on_single_ready:on_single_ready(single)
    else:
        single=saved_single;proposal=single['problems'];elapsed=single['seconds_to_single']
    t1=time.perf_counter()
    result=debate(doc,client,models,cfg,proposal,tag)
    duration=time.perf_counter()-t1
    multi={'paper_id':doc['paper_id'],'filename':doc['filename'],'mode':'multi',
           'problems':result['problems'],'rounds':result['rounds'],'best_score':result['best_score'],
           'best_round':result['best_round'],'stopping_reason':result['stopping_reason'],
           'final_decisions':result['final_decisions'],'history':result['history'],
           'seconds_debate':duration,'seconds_end_to_end':elapsed+duration,
           'resumed_from_saved_single':saved_single is not None}
    return single,multi
