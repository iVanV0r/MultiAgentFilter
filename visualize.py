"""Автономный HTML-отчёт: извлечение научных проблем, решения триады и ресурсы."""
from __future__ import annotations
import argparse
import csv
import html
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from settings import OUTPUT, EXCLUDED_PAPER_IDS


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def csv_rows(path):
    try:
        with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def esc(value):
    return html.escape(str(value if value is not None else ''), quote=True)


def num(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def fmt(value, digits=2, unit=''):
    n = num(value)
    return f'{n:.{digits}f}{unit}' if n is not None else '—'


def size_gb(bytes_value):
    n = num(bytes_value)
    return fmt(n / 1024**3, 2, ' ГБ') if n is not None else '—'


def badge(text, style='neutral'):
    return f'<span class="badge badge-{style}">{esc(text)}</span>'


def stat(title, value, subtitle=''):
    return (f'<div class="stat"><div class="stat-title">{esc(title)}</div>'
            f'<div class="stat-value">{esc(value)}</div><div class="stat-sub">{esc(subtitle)}</div></div>')


def evidence(items):
    if not items:
        return '<p class="empty">Цитата не указана.</p>'
    rows = []
    for e in items:
        if not isinstance(e, dict):
            continue
        v = e.get('verified')
        state = badge('Проверена по PDF', 'ok') if v is True else (badge('Не подтверждена автоматически', 'bad') if v is False else badge('Проверка не указана'))
        rows.append(f'<div class="quote"><div class="quote-top">Страница {esc(e.get("page", "?"))} {state}</div>'
                    f'<blockquote>{esc(e.get("quote", ""))}</blockquote></div>')
    return ''.join(rows) or '<p class="empty">Цитата не указана.</p>'


def evidence_invalid(problem):
    ev = problem.get('evidence') or []
    return (problem.get('grounded') is False or
            any(isinstance(e, dict) and e.get('verified') is False for e in ev))


def invalid_by_origin(single, multi):
    """Отказ автоматической проверки буквальной цитаты, не решение Gemma.

    Сырой Qwen и дополнения Phi/Qwen-защитника показываются у автора.
    Данные history.invalid_evidence_candidates содержат именно *дополнения Phi*,
    которые были проверены и исключены до рассмотрения Судьёй.
    """
    qwen, triad = [], []
    for event in (single or {}).get('chunk_events', []):
        for item in event.get('raw', []):
            if isinstance(item, dict) and evidence_invalid(item):
                qwen.append(dict(item, from_label=f'Qwen · черновой фрагмент {event.get("chunk", "?")}', origin_note='Не вошла в итоговый ответ Qwen: буквальная цитата или страница не прошла проверку.'))
    for item in (single or {}).get('problems', []):
        if isinstance(item, dict) and evidence_invalid(item):
            qwen.append(dict(item, from_label='Qwen · финальный ответ', origin_note='Цитата итоговой проблемы не прошла буквальную сверку.'))
    for rnd in (multi or {}).get('history', []):
        for item in rnd.get('invalid_evidence_candidates', []):
            if isinstance(item, dict):
                triad.append(dict(item, from_label=f'Phi · неподтверждённое дополнение, раунд {rnd.get("round", "?")}', origin_note='Предложено Критиком, не попало на рассмотрение Gemma из-за проверки цитаты.'))
        for item in rnd.get('defense_actions', []):
            if isinstance(item,dict) and item.get('action')=='invalid_evidence_addition':
                triad.append({'statement':item.get('statement',''),
                              'evidence':item.get('evidence',[]),'grounded':False,
                              'from_label':f'Qwen · дополнение при защите, раунд {rnd.get("round", "?")}',
                              'origin_note':'Предложено во время защиты, не попало на рассмотрение Gemma из-за проверки цитаты.'})
    def unique(items):
        result=[]; seen=set()
        for p in items:
            k=(p.get('statement','').strip().casefold(),
               json.dumps(p.get('evidence',[]), ensure_ascii=False,sort_keys=True))
            if k not in seen:
                result.append(p); seen.add(k)
        return result
    return unique(qwen),unique(triad)


def round_for(multi):
    history = (multi or {}).get('history', [])
    preferred = (multi or {}).get('best_round')
    for r in history:
        if r.get('round') == preferred:
            return r
    return history[-1] if history else {}


def lineage(single, multi):
    """Переход M1..Mn из Qwen в решение триады по сохранённым ID."""
    origins = (single or {}).get('problems', [])
    chosen = round_for(multi)
    decisions = {d.get('candidate_id'):d for d in (multi or {}).get('final_decisions', []) if isinstance(d,dict)}
    hist = (multi or {}).get('history', [])
    upto = (multi or {}).get('best_round')
    edits = defaultdict(list)
    for r in hist:
        if upto is not None and num(r.get('round')) is not None and num(r['round']) > upto:
            continue
        for a in r.get('defense_actions', []):
            if isinstance(a, dict) and a.get('candidate_id'):
                edits[a['candidate_id']].append(dict(a, round=r.get('round')))
    records = []
    # Старые серии, созданные ДО защиты по устойчивым ID, могли присвоить M1
    # совершенно другой проблеме. Не приписываем чужое решение исходному M1.
    legacy=bool(multi) and bool(origins) and not any('defense_actions' in r for r in hist)
    def norm(statement):
        return ' '.join(str(statement or '').casefold().split())
    for i, p in enumerate(origins, 1):
        cid=f'M{i}'; steps=edits.get(cid,[]); decision=decisions.get(cid)
        removed=next((a for a in steps if a.get('action')=='remove'),None)
        was_rewritten=any(a.get('action')=='rewrite' and a.get('old_statement')!=a.get('new_statement') for a in steps)
        legacy_note=''
        if legacy:
            exact=next((x for x in (multi or {}).get('final_decisions',[])
                       if norm(x.get('statement'))==norm(p.get('statement'))),None)
            accepted_exact=any(norm(x.get('statement'))==norm(p.get('statement'))
                               for x in (multi or {}).get('problems',[]))
            if accepted_exact:
                status='retained';decision=exact
                legacy_note='Сопоставлено по буквальному совпадению итоговой формулировки (старая серия без устойчивых ID).'
            elif exact and exact.get('status')=='rejected':
                status='rejected';decision=exact
            elif exact and exact.get('status')=='no_judge_decision':
                status='unknown';decision=exact
            else:
                status='lost';decision=None
                legacy_note='Старая серия: проблема исчезла из финального ответа, но отдельного решения по её исходному ID не записано.'
        elif decision and decision.get('status')=='accepted':
            status='rewritten' if was_rewritten else 'retained'
        elif removed:
            status='removed'
        elif decision and decision.get('status')=='rejected':
            status='rejected'
        elif decision and decision.get('status')=='no_judge_decision':
            status='unknown'
        else:
            status='unknown'  # Неполная история; статус нельзя достоверно установить
        records.append({'id':cid, 'original':p, 'status':status,'steps':steps,'decision':decision,
                        'reason':legacy_note or ((removed or {}).get('reason','') if removed else (decision or {}).get('reason','')),
                        'final_statement':(decision or {}).get('statement') or
                           (steps[-1].get('new_statement') if steps and steps[-1].get('action')!='remove' else '')})
    additions=[]
    for d in (multi or {}).get('final_decisions', []):
        if isinstance(d,dict) and d.get('candidate_id','').startswith(('N','P')):
            cid=d.get('candidate_id','')
            additions.append(dict(d, origin='Phi · Критик' if cid.startswith('P') else 'Qwen · Защита'))
    # Если в версии JSON дополнения есть в выбранном раунде, но Судья не сохранил ID,
    # не придумываем решение: показываем как неизвестное.
    return records, additions


def semantic_data(folder):
    base=folder/'evaluation_auto'
    summary=load(base/'summary.json',{}) or {}
    is_semantic=(summary.get('matching_method')=='semantic' or
                 summary.get('basis')=='automatic_multilingual_embedding_similarity_only')
    metrics={}
    matches={}
    if is_semantic:
        for row in csv_rows(base/'metrics.csv'):
            try: metrics[(row['paper_id'],int(row['repetition']),row['mode'])]=row
            except (KeyError,ValueError): continue
        for row in csv_rows(base/'matches.csv'):
            try: matches[(row['paper_id'],int(row['repetition']),row['mode'],row['prediction_id'])]=row
            except (KeyError,ValueError): continue
    return is_semantic, metrics, matches, num(summary.get('matching_threshold'))


def verified_semantic_matches(paper_id, rep, mode, problems, raw_matches, has_semantic):
    """Пары из оценки валидируются против текущего набора JSON.

    Старые результаты оценки могли быть вычислены для другого запуска при том
    же --name: не показываем произвольные числа для чужих формулировок.
    """
    aligned={}; mismatched=[]
    if not has_semantic:
        return aligned,mismatched
    for i,problem in enumerate(problems,1):
        key=(paper_id,rep,mode,f'M{i}')
        row=raw_matches.get(key)
        if row is None:
            mismatched.append(f'M{i}: нет записи в matches.csv')
            continue
        def normalize(t): return ' '.join(str(t or '').casefold().split())
        if normalize(row.get('statement'))!=normalize(problem.get('statement')):
            mismatched.append(f'M{i}: формулировка отличается от результатов оценки')
            continue
        aligned[f'M{i}']=row
    return aligned,mismatched


def article_similarity(aligned, problems):
    """Среднее max cos одной модели по каждой итоговой формулировке.

    Именно это же nearest_similarity выводится в каждой карточке проблемы.
    Если данных не хватает хотя бы для одной формулировки, среднее не выдаём.
    """
    if not problems:
        return None
    scores=[num(aligned.get(f'M{i}',{}).get('nearest_similarity'))
            for i in range(1,len(problems)+1)]
    if any(v is None for v in scores):
        return None
    return statistics.mean(scores)


def mean_by_articles(entries, key):
    """Каждая статья имеет одинаковый вес, даже если повторов разное число."""
    per_paper=defaultdict(list)
    for row in entries:
        value=num(row.get(key))
        if value is not None:
            per_paper[row['pid']].append(value)
    return (statistics.mean(statistics.mean(x) for x in per_paper.values()) if per_paper else None,
            len(per_paper))



def diagnostic_similarity(match, has_semantic):
    """Только диагностическое расстояние эмбеддингов. НЕ качество ответа."""
    if not has_semantic or not isinstance(match,dict):
        return ''
    sim = num(match.get('nearest_similarity'))
    return f'<span class="quiet-number">Косинусная близость: {fmt(sim,3)}</span>' if sim is not None else ''


def problem_card(problem, *, label='', style='', match=None, has_semantic=False, threshold=None,
                 decision='', reason='', revised='', show_status=True):
    proof_invalid=evidence_invalid(problem)
    simtext=diagnostic_similarity(match,has_semantic)
    head = (badge(label) if label else '')
    if decision and show_status:
        head += badge(decision, style or 'neutral')
        if proof_invalid: head += badge('Цитата не подтверждена', 'bad')
    final = (f'<div class="revision">Версия после обсуждения: {esc(revised)}</div>'
             if revised and revised.strip()!=problem.get('statement','').strip() else '')
    reason_html=f'<p class="reason">{esc(reason)}</p>' if reason else ''
    provenance_html=f'<p class="source-note">{esc(problem.get("origin_note",""))}</p>' if problem.get('origin_note') else ''
    sim_html=f'<div class="simtext">{simtext}</div>' if simtext else ''
    return (f'<article class="problem {esc(style)}"><div class="problem-head">{head}</div>'
            f'<div class="statement">{esc(problem.get("statement", ""))}</div>'
            f'{final}{reason_html}{provenance_html}{sim_html}'
            f'<details class="sub"><summary>Цитата из PDF</summary>{evidence(problem.get("evidence",[]))}</details></article>')


def unverified_cards(items):
    if not items: return ''
    return ('<details class="sub invalid-group"><summary>Черновые предложения с неподтверждённой цитатой '
            f'({len(items)})</summary><p class="panel-note">Буквальная сверка цитаты и страницы не удалась. Это может быть опечатка, OCR или неверная страница; такие черновики не входят в итоговые подтверждённые предложения.</p>'+
            ''.join(problem_card(p,label=p.get('from_label',''),style='bad',show_status=False)
                    for p in items)+'</details>')


def line_card(rec, match, is_semantic, threshold):
    status=rec['status']
    labels={'retained':('Сохранена триадой','ok'),'rewritten':('Переформулирована и принята','ok'),
            'rejected':('Отклонена Gemma','bad'),'removed':('Удалена Qwen после критики','bad'),
            'unknown':('Нет подтверждённого решения','warn'),
            'lost':('Исчезла без отдельного решения (старый формат)','warn')}
    title,style=labels[status]
    return problem_card(rec['original'],label=rec['id'],style=style,decision=title,reason=rec['reason'],
                        revised=rec['final_statement'] if status=='rewritten' else '',
                        match=match,has_semantic=False,threshold=None)


def system_metrics(single, multi):
    """Суммируем LLM вызовы, а не всю статью; пиковые сенсоры = max записанных samples."""
    stage={'Qwen · извлечение': [], 'Phi · критика': [], 'Qwen · защита':[], 'Gemma · арбитраж':[]}
    for event in (single or {}).get('chunk_events', []):
        if isinstance(event.get('metrics'),dict):stage['Qwen · извлечение'].append(event['metrics'])
    for metric in (single or {}).get('merge_metrics', []):
        if isinstance(metric,dict):stage['Qwen · извлечение'].append(metric)
    for rnd in (multi or {}).get('history', []):
        for metric in rnd.get('metrics', []):
            if not isinstance(metric,dict):continue
            role=metric.get('role','')
            if role=='critic': name='Phi · критика'
            elif role=='proposer_defense':name='Qwen · защита'
            elif role=='referee': name='Gemma · арбитраж'
            else: name='Gemma · арбитраж' if 'referee' in role else 'Qwen · защита'
            stage[name].append(metric)
    return stage


def summarize_stage(items):
    def summed(key): return sum(num(x.get(key)) or 0 for x in items)
    def peak(key):
        values=[num(x.get('hardware',{}).get(key)) for x in items]
        return max((v for v in values if v is not None),default=None)
    def mean_hardware(key):
        vals=[num(x.get('hardware',{}).get(key)) for x in items]
        vals=[v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None
    output=summed('generated_tokens')
    gen_secs=summed('eval_seconds')
    return {'calls':len(items), 'prompt_tokens':summed('prompt_tokens'),
            'generated_tokens':output,'wall_seconds':summed('wall_seconds'),
            'load_seconds':summed('load_seconds'), 'prompt_seconds':summed('prompt_eval_seconds'),
            'generation_seconds':gen_secs,'tps':output/gen_secs if gen_secs>0 else None,
            'retries':sum(1 for x in items if x.get('retried') or (num(x.get('attempt')) or 0)>0),
            'gpu_mb':peak('max_gpu_mem_mb'),'gpu_percent':peak('max_gpu_util_percent'),
            'ram_bytes':peak('max_system_ram_used_bytes'), 'cpu_percent':peak('max_cpu_percent'),
            'gpu_temp':peak('max_gpu_temperature_c'), 'gpu_power_w':peak('max_gpu_power_w'),
            'gpu_joules':sum(num(x.get('hardware',{}).get('estimated_gpu_joules')) or 0 for x in items),
            'models':sorted({x['model'] for x in items if x.get('model')}),
            'contexts':sorted({int(x['num_ctx']) for x in items if num(x.get('num_ctx')) is not None})}


def metric_table(single,multi):
    stages=system_metrics(single,multi)
    rows=[]
    for name,raw in stages.items():
        m=summarize_stage(raw)
        if not raw: continue
        rows.append(f'<tr><td><b>{esc(name)}</b><small>{esc(", ".join(m["models"]))}</small></td>'
                    f'<td>{m["calls"]}</td><td>{int(m["prompt_tokens"])} / {int(m["generated_tokens"])}</td>'
                    f'<td>{fmt(m["tps"],1)}</td><td>{fmt(m["wall_seconds"],1)} с</td>'
                    f'<td>{fmt(m["load_seconds"],1)} с</td><td>{fmt(m["generation_seconds"],1)} с</td>'
                    f'<td>{esc(" / ".join(str(x) for x in m["contexts"]))}</td>'
                    f'<td>{fmt(m["gpu_mb"],0)} МБ</td>'
                    f'<td>{fmt(m["ram_bytes"] / 1024**3 if m["ram_bytes"] is not None else None,1)} ГБ</td>'
                    f'<td>{fmt(m["gpu_percent"],0)}%</td><td>{fmt(m["cpu_percent"],0)}%</td>'
                    '</tr>')
    single_seconds=num((single or {}).get('seconds_to_single'))
    debate_seconds=num((multi or {}).get('seconds_debate'))
    total_seconds=num((multi or {}).get('seconds_end_to_end'))
    times=(f'<div class="runtime-strip">'
           f'<span>Qwen: <b>{fmt(single_seconds,1)} с</b></span>'
           f'<span>Дебаты дополнительно: <b>{fmt(debate_seconds,1)} с</b></span>'
           f'<span>Полная триада: <b>{fmt(total_seconds,1)} с</b></span>'
           f'<span>Раунды: <b>{esc((multi or {}).get("rounds", "—"))}</b></span>'
           f'<span>Gemma: <b>{esc((multi or {}).get("best_score", "—"))}/10</b></span></div>')
    if not rows:return times+'<p class="empty">Замеры инференса отсутствуют.</p>'
    return (times + '<div class="table-scroll"><table class="sys-table"><thead><tr><th>Модель и этап</th>'
            '<th>Запросы</th><th>Токены вход / выход</th><th>Ток/с</th><th>Вызовы</th>'
            '<th>Загрузка</th><th>Генерация</th><th>Контекст</th><th>VRAM пик</th><th>RAM пик</th>'
            '<th>GPU пик</th><th>CPU пик</th></tr></thead><tbody>'
            +''.join(rows)+'</tbody></table></div>'
            '<p class="footnote">Пики ресурсов измерены по выборкам мониторинга во время вызовов; '
            'энергия является приближённой оценкой, а время вызовов не включает весь Python-конвейер.</p>')


def history_html(multi):
    out=[]
    for rnd in (multi or {}).get('history',[]):
        critic=rnd.get('critic') or {}
        objections=critic.get('objections') or []
        actions=rnd.get('defense_actions') or []
        decisions=rnd.get('decisions') or []
        out.append(f'<details class="round"><summary>Раунд {esc(rnd.get("round","?"))} · '
                   f'оценка {esc(rnd.get("score","?"))}/10 · '
                   f'принято {sum(d.get("status")=="accepted" for d in decisions)}/{len(decisions)}</summary>'
                   '<div class="round-content"><h5>Phi — замечания</h5><ul>'+
                   (''.join(f'<li><b>{esc(o.get("target", "?"))}</b> {esc(o.get("criticism",""))}</li>' for o in objections)
                    or '<li>Нет замечаний</li>')+'</ul><h5>Qwen — действия</h5><ul>'+
                   (''.join(f'<li><b>{esc(a.get("candidate_id","?"))} · {esc(a.get("action",""))}:</b> {esc(a.get("reason",""))}</li>' for a in actions)
                    or '<li>Нет действий</li>')+'</ul><h5>Gemma — решения</h5><ul>'+
                   (''.join(f'<li><b>{esc(d.get("candidate_id","?"))} · {esc(d.get("status",""))}:</b> {esc(d.get("reason",""))}</li>' for d in decisions)
                    or '<li>Нет решений</li>')+'</ul></div></details>')
    return ''.join(out) or '<p class="empty">Раундов нет.</p>'


def selection_bar(entries):
    """Показывает последствия дебатов, а не недоказанное улучшение качества."""
    rows=[]
    for e in entries:
        q=e['initial'];m=e['accepted'];scale=max(q,m,1)
        label=esc(e['pid'])+(f' · #{e["rep"]}' if e['rep']!=1 else '')
        def track(value, cls):
            width=100*value/scale
            return f'<div class="bar-track"><span class="bar-fill {cls}" style="width:{width:.1f}%"></span></div>'
        rows.append(f'<div class="bar-line"><a href="#{esc(e["anchor"])}">{label}</a><div class="bars">'
                    f'{track(q,"qblue")}{track(m,"tgreen")}</div>'
                    f'<div class="bar-numbers">{q} / {m}</div></div>')
    return ''.join(rows) if rows else '<p class="empty">Пока нет результатов.</p>'



def report(folder, reviewed=False):
    """Генерация отчёта; параметр reviewed сохранён для совместимости с run.py."""
    folder=Path(folder)
    files={}
    for mode in ('single','multi'):
        for path in sorted((folder/mode).glob('*.json')):
            d=load(path)
            if isinstance(d,dict) and d.get('paper_id') and d['paper_id'] not in EXCLUDED_PAPER_IDS:
                files[(d['paper_id'],int(d.get('repetition',1)),mode)]=d
    keys=sorted({(pid,rep) for pid,rep,_ in files})
    has_semantic,metrics,matches,threshold=semantic_data(folder)
    summary=load(folder/'evaluation_auto'/'summary.json',{}) or {}
    matching_model=summary.get('embedding_model','') if has_semantic else ''
    entries=[]; papers=[]; total=Counter(); times=[]
    role_agg=defaultdict(list)
    for index,(pid,rep) in enumerate(keys,1):
        single=files.get((pid,rep,'single'));multi=files.get((pid,rep,'multi'))
        initial=(single or {}).get('problems',[])
        accepted=(multi or {}).get('problems',[])
        transitions,added=lineage(single,multi)
        states=Counter(x['status'] for x in transitions)
        valid_initial=states['retained']+states['rewritten']
        rejected=states['rejected']+states['removed']
        accepted_additions=sum(x.get('status')=='accepted' for x in added)
        unexpected=len(accepted)-valid_initial-accepted_additions
        q_bad,t_bad=invalid_by_origin(single,multi)
        aligned_single,issues_s=verified_semantic_matches(pid,rep,'single',initial,matches,has_semantic)
        aligned_multi,issues_m=verified_semantic_matches(pid,rep,'multi',accepted,matches,has_semantic)
        sim_single=article_similarity(aligned_single,initial)
        sim_multi=article_similarity(aligned_multi,accepted)
        if issues_s or issues_m:
            logging.warning('%s / повтор %s: оценка не соответствует JSON (%s; %s). Пересчитайте evaluate.py --matcher semantic',
                            pid, rep, ', '.join(issues_s[:2]), ', '.join(issues_m[:2]))
        sim_delta=sim_multi-sim_single if sim_single is not None and sim_multi is not None else None
        anchor=f'article-{index}'
        total.update({'qwen':len(initial),'accepted':len(accepted),'rejected':rejected,
                      'rewritten':states['rewritten'],'added':accepted_additions,
                      'bad_qwen':len(q_bad),'bad_triad':len(t_bad),'unknown':states['unknown'],
                      })
        seconds_s=num((single or {}).get('seconds_to_single'))
        seconds_m=num((multi or {}).get('seconds_end_to_end'))
        if seconds_s is not None and seconds_m is not None:times.append((seconds_s,seconds_m))
        stages=system_metrics(single,multi)
        for stage,rows in stages.items():
            role_agg[stage].extend(rows)
        filename=(single or multi or {}).get('filename',f'{pid}.pdf')
        entries.append({'pid':pid,'rep':rep,'filename':filename,'anchor':anchor,'initial':len(initial),
                        'kept':valid_initial,'rejected':rejected,'accepted':len(accepted),
                        'bad_qwen':len(q_bad),'bad_triad':len(t_bad),
                        'sim_single':sim_single,'sim_multi':sim_multi,'sim_delta':sim_delta,

                        'seconds_s':seconds_s,'seconds_m':seconds_m})
        q_cards=[line_card(rec,aligned_single.get(rec['id']),False,None) for rec in transitions]
        q_problems=('<div class="group-head"><h4>Qwen — первоначальные проблемы</h4>'
                    f'{badge(f"{len(initial)} найдено")} {badge(f"{rejected} отклонено/удалено", "bad")}'
                    '</div>' + (''.join(q_cards) if q_cards else '<p class="empty">Qwen не выделил проблем.</p>')
                    +unverified_cards(q_bad))
        final_dec=(multi or {}).get('final_decisions',[])
        accepted_dec=[d for d in final_dec if d.get('status')=='accepted']
        triad_cards=[]
        for i,p in enumerate(accepted,1):
            d=accepted_dec[i-1] if i-1<len(accepted_dec) else {}
            cid=d.get('candidate_id',f'#{i}')
            match=aligned_multi.get(f'M{i}')
            card=problem_card(p,label=cid,decision='Принята Gemma',style='ok',
                              reason=d.get('reason',''),has_semantic=False,threshold=None,
                              match=match)
            triad_cards.append(card)
        # Дополнения возможны на этапе Phi и Qwen-защиты; они не являются
        # повторным независимым запуском начального Qwen. Не выделяем их в отдельный KPI.
        addition_cards=[]
        for extra in added:
            is_accepted=extra.get('status')=='accepted'
            # Сходство для добавленного кандидата берём из соответствующей итоговой проблемы.
            ai=next((i for i,p in enumerate(accepted,1)
                     if ' '.join(str(p.get('statement','')).casefold().split())==
                        ' '.join(str(extra.get('statement','')).casefold().split())),None)
            extra_match=aligned_multi.get(f'M{ai}') if ai is not None else None
            addition_cards.append(problem_card(
                {'statement':extra.get('statement',''),'evidence':extra.get('evidence',[]),
                 'grounded':extra.get('grounded')},label=extra.get('origin',''),
                decision='Дополнение в дебатах · '+('принято' if is_accepted else 'отклонено'),
                style='ok' if is_accepted else 'bad',reason=extra.get('reason',''),
                match=extra_match,has_semantic=False,threshold=None,
                ))
        addition_html=''.join(addition_cards)
        triad_html=('<div class="group-head"><h4>Триада — итоговые проблемы</h4>'
                    f'{badge(f"{len(accepted)} принято", "ok")}</div>'
                    +(''.join(triad_cards) if triad_cards else '<p class="empty">Gemma не приняла ни одной проблемы.</p>')
                    +('<details class="sub"><summary>Дополнения, предложенные в дебатах '
                      f'({len(added)})</summary>{addition_html}</details>' if added else '')
                    +unverified_cards(t_bad))
        with_changes=''.join(f'<div class="transition {esc(rec["status"])}">'
                     f'<span class="transition-id">{esc(rec["id"])}</span>'
                     f'<span>{esc(rec["original"].get("statement",""))}</span>'
                     f'{badge({"retained":"Сохранена","rewritten":"Изменена","rejected":"Отклонена Gemma","removed":"Удалена Qwen","unknown":"Нет решения","lost":"Пропала без решения"}[rec["status"]],{"retained":"ok","rewritten":"teal","rejected":"bad","removed":"bad","unknown":"warn","lost":"warn"}[rec["status"]])}'
                     f'</div>' for rec in transitions)
        details=(f'<details class="paper" id="{anchor}" data-search="{esc((pid+" "+filename+" "+str(rep)).casefold())}">'
                 f'<summary><div><strong>{esc(filename)}</strong><small>Повтор {rep} · '
                 f'Qwen {len(initial)} → триада {len(accepted)} · отклонено/удалено {rejected}</small></div>'
                 f'<span class="paper-sim">Развернуть <span class="down">⌄</span></span></summary>'
                 f'<div class="paper-content"><div class="lineage-block"><h4>Судьба исходных проблем Qwen</h4>'
                 f'{with_changes or "<p class=\"empty\">Исходных проблем нет.</p>"}</div>'
                 f'<div class="two-col"><section>{q_problems}</section><section>{triad_html}</section></div>'
                 f'<div class="paper-bottom"><details class="sub"><summary>Время, GPU, RAM и токены по моделям</summary>'
                 f'{metric_table(single,multi)}</details>'
                 f'<details class="sub"><summary>История критики, защиты и решений</summary>{history_html(multi)}</details>'
                 f'</div></div></details>')
        papers.append(details)

    table=[]
    for e in entries:
        table.append(f'<tr data-search="{esc((e["pid"]+" "+e["filename"]+" "+str(e["rep"])).casefold())}">'
                     f'<td><a href="#{esc(e["anchor"])}">{esc(e["pid"])}'+(f' <small>#{e["rep"]}</small>' if e['rep']!=1 else '')+'</a></td>'
                     f'<td>{e["initial"]}</td><td>{e["kept"]}</td>'
                     f'<td>{e["bad_qwen"]} / {e["bad_triad"]}</td>'
                     f'<td>{fmt(e["seconds_s"],1)} / {fmt(e["seconds_m"],1)}</td></tr>')
    avg_time_s=statistics.mean(s for s,_ in times) if times else None
    avg_time_m=statistics.mean(m for _,m in times) if times else None
    session_stages={name:summarize_stage(items) for name,items in role_agg.items()}
    requests_total=sum(v['calls'] for v in session_stages.values())
    retries_total=sum(v['retries'] for v in session_stages.values())
    report_stats=''.join([
        stat('Обработано',sum((pid,rep,'single') in files and (pid,rep,'multi') in files for pid,rep in keys),
             'статей'),
        stat('Нашёл Qwen',total['qwen'],'Исходные итоговые кандидаты'),
        stat('Триада приняла',total['accepted'],'Итоговые формулировки'),
        stat('Триада отклонила',total['rejected'],'Отклонено судьей или генератором в процессе обсуждения'),
        stat('Триада изменила',total['rewritten'],'Приняты после переформулирования'),
    ])
    css=r'''
:root{color-scheme:light;--bg:#f4f6fa;--ink:#1c2a40;--muted:#64748b;--border:#e2e7ef;--white:#fff;--blue:#2c6bcd;--green:#168579;--red:#ac4451}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{background:var(--bg);margin:0;font:14px/1.56 system-ui,-apple-system,"Segoe UI",Arial,sans-serif;color:var(--ink)}a{color:#2963a9;text-decoration:none}a:hover{text-decoration:underline}
.header{background:#14283f;color:#fff;padding:35px 24px}.header-inner{max-width:1250px;margin:auto}.header .overline{font-size:11px;font-weight:800;letter-spacing:.16em;color:#9bc3e7}.header h1{font-size:31px;letter-spacing:-.035em;margin:5px 0 8px}.header p{color:#cbd8e7;margin:0}.nav{display:flex;gap:25px;margin-top:18px}.nav a{font-size:12px;color:#e5f4ff}.container{max-width:1290px;margin:0 auto;padding:22px 20px 70px}.section{margin-bottom:26px}.section-header{display:flex;justify-content:space-between;gap:18px;align-items:end;margin-bottom:12px;flex-wrap:wrap}.section-header h2{font-size:19px;letter-spacing:-.015em;margin:0}.section-header p{color:var(--muted);margin:0;font-size:12px}
.stat-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.stat{background:#fff;border:1px solid var(--border);border-radius:11px;padding:16px 14px}.stat-title{font-size:11px;color:var(--muted);min-height:31px}.stat-value{font-size:26px;letter-spacing:-.04em;font-weight:770}.stat-sub{font-size:10px;color:var(--muted);margin-top:5px}
.two-panels{display:grid;grid-template-columns:1.2fr 1fr;gap:15px}.panel,.table-scroll{border:1px solid var(--border);border-radius:11px;background:#fff;padding:18px 20px}.panel h3{font-size:15px;margin:0 0 12px}.resource-panel>summary{cursor:pointer;font-weight:760;font-size:14px}.resource-panel .table-scroll{margin-top:15px}.panel-note{font-size:12px;color:var(--muted)}.chart-means{display:flex;flex-wrap:wrap;gap:10px;margin:9px 0 14px}.chart-means span{padding:7px 11px;background:#f1f6fd;border-radius:8px;font-size:12px}.chart-means b{font-size:16px;color:#123d6c}.chart-means small{color:#64748b}.source-note{color:#5e7287;font-size:11px;background:#f4f7fb;padding:7px 9px;border-radius:6px}.numbers{display:grid;grid-template-columns:1fr 1fr;gap:10px}.number-box{background:#f4f7fb;padding:13px;border-radius:9px}.number-box b{display:block;font-size:22px}.number-box span{font-size:11px;color:var(--muted)}
.chart-toggle{display:flex;gap:5px;padding:4px;background:#edf2f8;border-radius:8px;width:fit-content;margin-bottom:13px}.chart-toggle button{background:transparent;border:0;border-radius:6px;padding:6px 11px;color:#52657b;font:inherit;font-weight:650;font-size:11px;cursor:pointer}.chart-toggle button.selected{background:#fff;color:#163d69;box-shadow:0 1px 3px #0002}[hidden]{display:none!important}.chart-key{display:flex;gap:15px;font-size:11px;color:var(--muted);margin:0 0 12px}.legend{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}.qblue{background:var(--blue)}.tgreen{background:var(--green)}.chart{max-height:350px;overflow:auto}.bar-line{display:grid;grid-template-columns:125px 1fr 96px;gap:11px;align-items:center;margin:10px 0;font-size:11px}.bar-line>a{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bars{display:flex;flex-direction:column;gap:4px}.bar-track{height:8px;background:#eef1f6;border-radius:7px;overflow:hidden}.bar-fill{display:block;height:100%;border-radius:7px}.bar-numbers{text-align:right;font-variant-numeric:tabular-nums}
.table-scroll{overflow-x:auto;padding:0}.score-table,.sys-table{border-collapse:collapse;width:100%;min-width:1000px}.score-table th,.sys-table th{background:#f2f5f9;color:#586982;text-align:left;font-size:11px;padding:12px 10px;white-space:nowrap}.score-table td,.sys-table td{border-top:1px solid var(--border);padding:10px;font-size:12px;white-space:nowrap;font-variant-numeric:tabular-nums}.score-table td:first-child{font-weight:700}.score-table tr:hover{background:#f8fbff}.sys-table small{display:block;color:var(--muted);font-weight:400;font-size:10px}.color-blue{color:#2d61b4;font-weight:700}.color-green{color:#0a7b6f;font-weight:700}.rise{color:#08765a;font-weight:800}.fall{color:#b33b56;font-weight:800}
.search{border:1px solid #cbd6e6;padding:11px 14px;border-radius:8px;background:#fff;font:inherit;color:var(--ink);width:min(440px,100%);margin-bottom:12px}.paper{background:#fff;border:1px solid var(--border);border-radius:11px;overflow:hidden;margin-bottom:10px;scroll-margin-top:18px}.paper>summary{cursor:pointer;list-style:none;padding:17px 18px;display:flex;align-items:center;justify-content:space-between;gap:18px}.paper>summary::-webkit-details-marker{display:none}.paper>summary strong{display:block;font-size:13px}.paper>summary small{color:var(--muted);font-size:11px}.paper-sim{color:var(--muted);font-size:11px;white-space:nowrap}.down{padding-left:10px;font-size:19px}.paper[open]>summary{border-bottom:1px solid var(--border)}.paper-content{padding:16px 18px 20px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:19px}.two-col>section{min-width:0}.group-head{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:0 0 10px}.group-head h4{font-size:14px;margin:0 auto 0 0}
.badge{display:inline-block;border-radius:24px;background:#eef2f7;color:#43566c;padding:3px 9px;font-size:10px;line-height:1.4;font-weight:720;white-space:normal}.badge-ok{background:#e2f5ee;color:#176b50}.badge-teal{background:#dcf4f0;color:#116f68}.badge-bad{background:#ffedf0;color:#a63b54}.badge-warn{background:#fff0d8;color:#926022}
.lineage-block{background:#f5f8fc;border:1px solid #e6edf4;border-radius:9px;padding:13px 15px;margin-bottom:16px}.lineage-block h4{margin:0 0 9px;font-size:13px}.transition{display:grid;grid-template-columns:37px 1fr auto;align-items:start;gap:11px;border-top:1px solid #e6edf4;padding:8px 0;font-size:12px}.transition-id{font-weight:800;color:#50627a}.transition>span:nth-child(2){overflow-wrap:anywhere}
.problem{background:#fff;color:var(--ink);border:1px solid #e2e8f0;border-radius:9px;padding:13px;margin-bottom:10px}.problem.bad{background:#fff9f9;border-color:#efcfd7}.problem.warn{background:#fffaf1;border-color:#eddfc9}.problem.ok{border-left:3px solid #70c3ac}.problem-head{display:flex;gap:5px;flex-wrap:wrap}.statement{font-weight:660;font-size:13px;color:var(--ink);margin:9px 0;overflow-wrap:anywhere}.reason{font-size:12px;color:#526074;margin:7px 0}.revision{background:#f2f7fc;border-radius:7px;padding:9px 10px;color:#344d67;margin:7px 0;font-size:12px}.simtext{color:#596d81;font-size:11px;margin:6px 0}.sub{border-top:1px solid #edf1f6;margin-top:10px}.sub>summary{cursor:pointer;padding:8px 0;font-size:12px;color:#2e619a;font-weight:700}.invalid-group{margin:14px 0}.quote{margin:10px 0}.quote-top{display:flex;gap:7px;align-items:center;font-size:11px;color:#61728a}.quote blockquote{color:#294057;background:#f2f6fa;border-left:3px solid #adc3d8;padding:10px 12px;margin:6px 0;overflow-wrap:anywhere;font-size:12px}.empty{font-size:12px;color:var(--muted)}.paper-bottom{border-top:1px solid var(--border);margin-top:15px;padding-top:10px}.runtime-strip{background:#f5f8fc;border-radius:8px;padding:11px;display:flex;flex-wrap:wrap;gap:16px;font-size:12px;margin:8px 0}.unmatched-group{padding:8px 11px;background:#fff7f7;border:1px solid #f2d1d7;border-radius:9px;margin:14px 0}.unmatched-group>summary{color:#9c3346;font-weight:770}.unmatched-group .problem{margin-top:9px}.sys-table{min-width:1380px}.round{border:1px solid var(--border);border-radius:7px;margin:6px 0}.round>summary{cursor:pointer;padding:9px 11px;font-size:12px}.round-content{padding:0 14px 12px;font-size:12px}.round-content h5{margin:10px 0 4px}.round-content ul{padding-left:19px}.round-content li{margin:5px 0;overflow-wrap:anywhere}.footnote{font-size:10px;color:#718197}.notice{font-size:11px;color:#66768b;margin-top:12px}.quiet-number{font-size:10px;color:#768397}
@media(max-width:1250px){.stat-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:850px){.stat-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.two-col,.two-panels{grid-template-columns:1fr}.container{padding:15px 11px 55px}.header{padding:24px 12px}.header h1{font-size:25px}.bar-line{grid-template-columns:96px 1fr 75px}.paper>summary{flex-wrap:wrap}.paper-sim{white-space:normal}.transition{grid-template-columns:30px 1fr}.transition>.badge{grid-column:2}.panel{padding:14px}}
'''
    js="""(()=>{const input=document.querySelector('#search');const count=document.querySelector('#shown-count');function filter(){const value=input.value.toLocaleLowerCase().trim();let n=0;document.querySelectorAll('details.paper, #comparison-body tr').forEach(el=>{const visible=el.dataset.search.includes(value);el.hidden=!visible;if(visible&&el.matches('details.paper'))n++});count.textContent=n;}input.addEventListener('input',filter);filter();})();"""
    global_rows=[]
    for name in ('Qwen · извлечение','Phi · критика','Qwen · защита','Gemma · арбитраж'):
        m=session_stages.get(name)
        if not m or not m['calls']: continue
        global_rows.append(f'<tr><td><b>{esc(name)}</b><small>{esc(", ".join(m["models"]))}</small></td>'
                           f'<td>{m["calls"]}</td><td>{int(m["prompt_tokens"])} / {int(m["generated_tokens"])}</td>'
                           f'<td>{fmt(m["tps"],1)}</td><td>{fmt(m["wall_seconds"],1)} с</td>'
                           f'<td>{fmt(m["load_seconds"],1)} с</td><td>{fmt(m["gpu_mb"],0)} МБ</td>'
                           f'<td>{fmt(m["ram_bytes"] / 1024**3 if m["ram_bytes"] is not None else None,1)} ГБ</td>'
                           f'<td>{fmt(m["gpu_percent"],0)}%</td></tr>')
    page=(f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
          f'<title>Выявление научных проблем — Qwen и триада</title><style>{css}</style></head><body>'
          f'<header class="header"><div class="header-inner">'
          f'<h1>Многоагентная фильтрация научных проблем в полнотекстовых публикациях: парный анализ локальных SLM-архитектур</h1><p>Одиночная Qwen и триада Qwen–Phi–Gemma</p>'
          f'<nav class="nav"><a href="#overview">Сводка</a><a href="#papers">По статьям</a>'
          f'<a href="#resources">Ресурсы</a><a href="#problems">Проблемы и решения</a></nav></div></header><main class="container">'
          f'<section class="section" id="overview"><div class="section-header"><h2>Результаты отбора</h2>'
          f'</div><div class="stat-grid">{report_stats}</div></section>'
          f'<section class="section"><div class="two-panels"><div class="panel"><h3>Количество проблем по статьям</h3>'
          f'<div class="chart-key"><span><i class="legend qblue"></i>Qwen</span><span><i class="legend tgreen"></i>Триада</span></div>'
          f'<div class="chart" id="chart-selection">{selection_bar(entries)}</div>'
          f'</div>'
          f'<div class="panel"><h3>Вычислительная стоимость</h3>'
          f'<div class="numbers">'
          f'<div class="number-box"><b>{fmt(avg_time_s,1)} с</b><span>Qwen · время на статью</span></div>'
          f'<div class="number-box"><b>{fmt(avg_time_m,1)} с</b><span>Триада · время на статью</span></div>'
          f'<div class="number-box"><b>{requests_total}</b><span>Вызовов моделей</span></div>'
          f'<div class="number-box"><b>{retries_total}</b><span>Повторных запросов моделей в результате ошибки</span></div>'
          f'</div>'

          f'</div></div></section>'
          f'<section class="section" id="papers"><div class="section-header"><h2>Сравнение по публикациям</h2>'
          f'</div>'
          f'<div class="table-scroll"><table class="score-table"><thead><tr>'
          f'<th>Статья</th><th>Qwen нашёл</th><th>Триада сохранила</th>'
          f'<th>Цитата не подтверждена<br>Qwen / триада</th><th>Время Qwen / триада, с</th>'
          f'</tr></thead><tbody id="comparison-body">'+(''.join(table) if table else '<tr><td colspan="5">Данных пока нет.</td></tr>')
          +'</tbody></table></div></section>'
          f'<section class="section" id="resources"><details class="panel resource-panel"><summary>Системные показатели по всем моделям и ролям</summary>'
          f'<div class="table-scroll"><table class="sys-table"><thead><tr><th>Роль</th><th>Вызовы</th><th>Токены вход/выход</th>'
          f'<th>Ток/с</th><th>Время вызовов</th><th>Загрузка</th><th>VRAM пик</th><th>RAM пик</th><th>GPU пик</th>'
          f'</tr></thead><tbody>{"".join(global_rows)}</tbody></table></div>'
          f'<p class="footnote">Измерения агрегированы по записанным вызовам. Средние скорости рассчитываются по токенам и времени генерации.</p>'
          f'</details></section>'
          f'<section class="section" id="problems"><div class="section-header"><h2>Результаты обработки по статьям</h2>'
          f'</div>'
          f'<input id="search" class="search" type="search" placeholder="Найти статью или повтор" aria-label="Поиск по статьям">'
          f'<div class="notice"><span id="shown-count">{len(papers)}</span> обработок</div>'
          +(''.join(papers) if papers else '<div class="panel">После запуска моделей здесь появятся результаты.</div>')+
          '</section></main><script>'+js+'</script></body></html>')
    dest=folder/('report_reviewed.html' if reviewed else 'report.html')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(page,encoding='utf-8')
    logging.info('HTML: %s; %d статей × повторов; семантика=%s',dest,len(keys),has_semantic)
    return dest


def build_report(folder,reviewed=False):
    return report(folder,reviewed)


def main():
    ap=argparse.ArgumentParser(description='HTML отчёт: извлечение проблем, решения триады и ресурсы')
    ap.add_argument('--name',default='full')
    ap.add_argument('--reviewed',action='store_true',help='Совместимость с прежним CLI; отдельная карточка качества не выводится')
    a=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(message)s')
    report(OUTPUT/a.name,a.reviewed)


if __name__=='__main__':
    main()
