"""Только промпты; изменение этого файла требует новой серии --name."""
import json

DEFINITION = (
    'Научная проблема — явно или косвенно обозначенный пробел в знаниях либо недостаток '
    'существующих методов, данных или инструментов, на устранение которого НАПРАВЛЕНА работа. '
    'Цель, тема, описание решения, собственное ограничение проведённого эксперимента или '
    'упоминание чужой проблемы БЕЗ связи с целью статьи не являются научными проблемами. '
    'Допустимо ноль проблем. Формулировки на русском, цитаты дословно на языке источника.'
)
SCHEMA = ('{"problems":[{"statement":"исследовательский пробел",'
          '"type":"knowledge|method|data|tool", "explicitness":"explicit|implicit",'
          '"evidence":[{"page":1,"quote":"дословная фраза из PDF"}]}]}')
SYSTEM = {
    'proposer': DEFINITION+'\nОтвет — один КОРОТКИЙ валидный JSON, без Markdown.',
    'critic': DEFINITION+'\nКритикуй научную точность, а не качество написания статьи. Только JSON.',
    'referee': DEFINITION+'\nОцени КАЖДОГО кандидата, сохрани его candidate_id. Только JSON.'
}

def j(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))

def proposer_chunk(chunk, max_candidates):
    return (f'Найди 0–{max_candidates} различных научных проблем, направленных на цель ДАННОЙ работы. '
            'Не называй ограничение авторского эксперимента исходной научной проблемой. '
            'Не выдумывай дословные цитаты; используй SOURCE PAGE. Пиши коротко. '
            f'Формат {SCHEMA}\nИСТОЧНИК:\n{chunk}')

def proposer_merge(items):
    return ('Объедини ТОЛЬКО истинные смысловые дубли, НЕ сокращай список разных проблем. '
            'Верни максимум столько, сколько входных элементов. Сохрани исходные цитаты и страницы, '
            'не придумывай новые. Если не уверена в дубле — сохрани отдельно. '
            f'Ответ: {SCHEMA}\nКАНДИДАТЫ: {j(items)}')

def proposer_final(items, limit):
    return (f'Составь независимый ответ Qwen: максимум {limit} разных проблем. '
            'Приоритет: основная проблема, затем независимые вспомогательные. '
            'Удаляй лишь точные дубли и неподтверждённые темы, не превращай недостатки '
            'второстепенного эксперимента в главную проблему. Сохраняй цитаты как есть. '
            f'Формат: {SCHEMA}\nКАНДИДАТЫ: {j(items)}')

def critic(context, proposals, previous):
    small=[{'candidate_id':p['candidate_id'],'statement':p['statement'],
            'evidence':p.get('evidence',[])[:2]} for p in proposals]
    return ('Проверь каждую научную проблему КАК РЕЦЕНЗЕНТ. До 3 замечаний, но не придумывай их. '
            'Укажи target строго по candidate_id. Критика должна выявить: неверный пробел, '
            'проблему не этой статьи, недоказанность, дубли или слишком широкую формулировку. '
            'Дополнительно НАЙДИ ПРОПУЩЕННЫЕ проблемы, непосредственно решаемые авторами, '
            'и представь их как missed_problems с ДОСЛОВНОЙ цитатой и номером SOURCE PAGE. '
            'Если доказательства нет, missed_problems пустые. '
            'Формат: {"objections":[{"target":"M1","criticism":"почему",'
            '"source_page":1,"support_quote":"фрагмент или пусто"}],'
            '"missed_problems":[]}\n'
            'Пример missed_problems: [{"statement":"...","type":"method","explicitness":"explicit",'
            '"evidence":[{"page":2,"quote":"цитата"}]}].\n'
            f'КАНДИДАТЫ: {j(small)}\nПРЕДЫДУЩЕЕ: {j(previous)}\n'
            f'КОНТЕКСТ ПО ВСЕЙ СТАТЬЕ: {context}')

def defense(context, proposals, critique, round_no):
    small=[{'candidate_id':p['candidate_id'],'statement':p['statement'],
            'evidence':p.get('evidence',[])[:2]} for p in proposals]
    return (f'Раунд {round_no}. Для КАЖДОГО исходного candidate_id верни ровно одно действие: '
            'keep, rewrite или remove. Пропуск действия НЕ означает удаление. '
            'remove возможен только когда проблема НЕ соответствует определению или статье; '
            'обязательно конкретное обоснование. rewrite меняет ТОЛЬКО statement; '
            'исходные подтверждённые цитаты сохраняются. Для дополнений критика '
            'используй отдельное поле additions с ДОСЛОВНЫМИ цитатами и SOURCE PAGE. '
            'Не создавай больше двух дополнений. Ответ строго JSON: '
            '{"actions":[{"candidate_id":"M1","action":"keep",'
            '"statement":"исправленная фраза только если rewrite","reason":"аргумент"}],'
            '"additions":[{"statement":"...","type":"method",'
            '"explicitness":"explicit","evidence":[{"page":1,"quote":"цитата"}]}]}. '
            'Вывод краткий. '
            f'ИСХОДНЫЕ ПРОБЛЕМЫ: {j(small)}\nКРИТИКА: {j(critique)}\nКОНТЕКСТ: {context}')

def referee(context, proposals, critique, threshold):
    small=[{'candidate_id':p['candidate_id'],'statement':p['statement'],
            'evidence':p.get('evidence',[])[:2]} for p in proposals]
    return ('Независимо оцени ВСЕ candidate_id. Критерии раунда: научный пробел 0–3, '
            'связь с целью 0–2, цитата 0–3, конкретность 0–2. '
            'Высокий балл НЕ делает ложные утверждения правильными. '
            'accept=true ТОЛЬКО если проблема решается или систематизируется ДАННОЙ публикацией, '
            'а не является оговоркой о недостатках собственного эксперимента. '
            'По КАЖДОМУ ID обязательно одно решение. Если не уверен — accept=false. '
            'Ответ JSON: {"rubric":{"gap":3,"relevance":2,"evidence":3,"precision":2},'
            '"score":10,"reason":"кратко",'
            '"decisions":[{"candidate_id":"M1","accept":true,"reason":"почему"}]}. '
            f'КАНДИДАТЫ: {j(small)}\nКРИТИКА: {j(critique.get("objections",[])[:3])}\n'
            f'КОНТЕКСТ: {context}')

def referee_missing(missing,context):
    return ('Предыдущий ответ судьи содержал НЕПОЛНЫЕ решения. Верни решения ТОЛЬКО '
            'для всех указанных candidate_id, без score/rubric. '
            'Строго JSON: {"decisions":[{"candidate_id":"M1","accept":false,"reason":"кратко"}]}. '
            'Не пропускай ID. Если неизвестно — accept=false и обоснуй. '
            f'КАНДИДАТЫ: {j(missing)}\nКОНТЕКСТ: {context[:1800]}')
