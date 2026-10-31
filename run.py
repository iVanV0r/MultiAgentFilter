"""Запуск парного эксперимента: Qwen самостоятельно / Qwen+Phi+Gemma."""
import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agents import run_paper
from ollama_api import OllamaClient
from prepare import extract_pdf, write_json
from settings import INPUT_PDFS, HUMAN_GOLD, MODELS, OLLAMA_URL, OPTIONS, OUTPUT, REPEATS, ROOT, SIMILARITY_THRESHOLD, EVALUATION_MATCHER, EMBEDDING_MODEL, EXCLUDED_PAPER_IDS


def configure_logging(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    file_handler = logging.FileHandler(folder / 'experiment.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_models(client):
    try:
        models = client.list_models()
    except Exception as exc:
        raise SystemExit(f'Ollama недоступна по {OLLAMA_URL}: {exc}\nЗапустите Ollama и повторите.')
    for role, name in MODELS.items():
        status = 'OK' if name in models else 'НЕТ'
        logging.info('Модель %-8s | %-33s | %s', role, name, status)
    missing = [name for name in MODELS.values() if name not in models]
    if missing:
        raise SystemExit('Отсутствуют модели. Скачайте через ollama pull: ' + ', '.join(missing))
    return {name: models[name] for name in MODELS.values()}


def source_docs(files):
    processed = OUTPUT / 'processed'
    processed.mkdir(parents=True, exist_ok=True)
    docs = []
    for n, pdf in enumerate(files, 1):
        target = processed / (pdf.stem + '.json')
        current_hash = sha256(pdf)
        if target.exists():
            try:
                data = json.loads(target.read_text(encoding='utf-8'))
                wanted = {k:OPTIONS[k] for k in ('chunk_chars','overlap_chars','include_references')}
                if data.get('sha256') != current_hash or data.get('parse_config')!=wanted:
                    logging.info('[%s/%s] PDF изменён, обрабатываю заново: %s', n, len(files), pdf.name)
                    data = None
            except (ValueError, OSError):
                data = None
        else:
            data = None
        if data is None:
            logging.info('[%s/%s] Извлечение всех страниц: %s', n, len(files), pdf.name)
            started = time.perf_counter()
            data = extract_pdf(pdf, OPTIONS['chunk_chars'], OPTIONS['overlap_chars'], OPTIONS['include_references'])
            data['parse_seconds'] = time.perf_counter()-started
            write_json(target, data)
        docs.append(data)
        logging.info('  Подготовлено: страниц=%s | фрагментов=%s | язык=%s', data['pdf_pages'], len(data['chunks']), data['language'])
    return docs


def environment_snapshot(models, files, repeats, ollama_version):
    import psutil
    from importlib.metadata import version, PackageNotFoundError
    pkgs = {}
    for name in ('PyMuPDF','requests','psutil','numpy','scipy','scikit-learn','matplotlib'):
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    gpu = []
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,timeout=8,check=True)
        gpu = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    except (FileNotFoundError,subprocess.SubprocessError):
        pass
    return {
        'gpu_devices':gpu,
        'ollama_version':ollama_version,
        'run_at_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'python': sys.version,
        'ram_total_gb': round(psutil.virtual_memory().total/(1024**3),2),
        'ollama_url': OLLAMA_URL,
        'ollama_models': models,
        'models_by_role': MODELS,
        'options': OPTIONS,
        'repeats': repeats,
        'packages': pkgs,
        'pdf_hashes': {p.name:sha256(p) for p in files},
        'human_gold_sha256': sha256(HUMAN_GOLD),
        'prompts_sha256': sha256(ROOT/'prompts.py'),
        'pipeline_sha256': {name:sha256(ROOT/name) for name in ('agents.py','ollama_api.py','prepare.py')},
    }


def run(args):
    if not args.name.replace('_','').replace('-','').isalnum():
        raise SystemExit('Имя серии должно содержать только буквы, цифры, _ или -.')
    folder = OUTPUT / args.name
    configure_logging(folder)
    all_files = sorted(p for p in INPUT_PDFS.glob('*.pdf') if p.stem not in EXCLUDED_PAPER_IDS)
    files = all_files
    if args.paper:
        files = [p for p in files if args.paper.casefold() in p.name.casefold()]
    files = files[args.start:]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'По параметрам отбора PDF не найдены в {INPUT_PDFS}')
    logging.info('SciGap | выбрано PDF=%s из %s | пропущено с начала=%s | лимит=%s | повторов=%s | всего парных запусков=%s',
                 len(files), len(all_files), args.start, args.limit or 'все', args.repeats, len(files)*args.repeats)
    logging.info('Отобранные статьи: %s', ', '.join(p.name for p in files))

    logging.info('Вариант A: Qwen; вариант B: Qwen -> Phi -> Qwen -> Gemma, до %s раундов', OPTIONS['max_rounds'])
    client = OllamaClient(OLLAMA_URL, OPTIONS, folder/'telemetry.jsonl')
    models_info = installed_models(client)
    ollama_version = client.version()
    logging.info('Версия Ollama: %s', ollama_version)
    if args.check:
        logging.info('Подключение и модели проверены. Для эксперимента запустите без --check.')
        return
    docs = source_docs(files)
    # Сохраняем отпечаток ВСЕГО фиксированного корпуса, даже если в текущем запуске выбрана только часть статей.
    # Это позволяет обрабатывать одну серию партиями с одинаковым значением --name.
    snapshot = environment_snapshot(models_info, all_files, args.repeats, ollama_version)
    selection_log = folder/'selection_history.jsonl'
    with selection_log.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                            'start':args.start,'limit':args.limit,'paper_filter':args.paper,
                            'selected_files':[p.name for p in files]}, ensure_ascii=False)+'\n')
    meta = folder / 'environment.json'
    if meta.exists():
        old = json.loads(meta.read_text(encoding='utf-8'))
        fields = ('ollama_models','models_by_role','options','repeats','pdf_hashes','human_gold_sha256','prompts_sha256','pipeline_sha256','ollama_version')
        mismatch = [k for k in fields if old.get(k)!=snapshot.get(k)]
        if mismatch:
            # При исправлении парсинга JSON можно восстановить почти законченный запуск.
            # Смешанные конфигурации нельзя выдавать за строгий единый эксперимент.
            immutable=('ollama_models','models_by_role','repeats','pdf_hashes','human_gold_sha256','prompts_sha256')
            unsafe=[k for k in immutable if old.get(k)!=snapshot.get(k)]
            # В старых версиях проекта хеши рассчитывались только для выбранной партии.
            # Расширять охват партии безопасно, если все файлы с ранее сохранёнными хешами не изменились.
            old_hashes=old.get('pdf_hashes') or {}
            new_hashes=snapshot['pdf_hashes']
            if 'pdf_hashes' in unsafe and old_hashes and all(
                    name in new_hashes and digest==new_hashes[name]
                    for name,digest in old_hashes.items()):
                unsafe.remove('pdf_hashes')
                logging.warning('Предыдущая версия хранила хеши выбранной партии; весь корпус теперь проверяется целиком.')
            if not args.resume_after_fix or unsafe:
                raise SystemExit(
                    'Конфигурация сохранённого запуска отличается: '+', '.join(mismatch)+'. '
                    + ('Изменились неизменяемые условия: '+', '.join(unsafe)+'. ' if unsafe else '')
                    + 'Для чистого эксперимента используйте --name новое_имя. '
                    + ('Для продолжения прежних результатов после исправления JSON добавьте --resume-after-fix.' if not unsafe else '')
                )
            warning='ВНИМАНИЕ: возобновление серии с изменёнными настройками/кодом! Результаты серии неоднородны для научного сравнения.'
            logging.warning(warning)
            update={'utc':datetime.now(timezone.utc).isoformat(), 'changed_fields':mismatch,
                    'old_environment':old,'new_environment':snapshot,'warning':warning}
            history=folder/'config_changes.jsonl'
            with history.open('a',encoding='utf-8') as f:
                f.write(json.dumps(update,ensure_ascii=False)+'\n')
            logging.warning('Изменения записаны: %s',history)
        else:
            logging.info('Найдена серия с теми же настройками. Применяю возобновление.')
    else:
        write_json(meta, snapshot)
    total = len(docs)*args.repeats
    failures = []
    begin = time.perf_counter()
    completed = 0
    try:
        for repeat in range(1,args.repeats+1):
            for doc in docs:
                completed += 1
                key = f'{doc["paper_id"]}__rep{repeat}'
                single_file = folder/'single'/f'{key}.json'
                multi_file = folder/'multi'/f'{key}.json'
                logging.info('=== [%s/%s] %s | повтор %s/%s | фрагментов %s ===', completed,total,doc['filename'],repeat,args.repeats,len(doc['chunks']))
                if single_file.exists() and multi_file.exists():
                    logging.info('Пропущено: результаты обоих режимов уже записаны.')
                    continue
                if OPTIONS.get('reset_between_papers',True):
                    client.unload()
                try:
                    pdf = INPUT_PDFS/doc['filename']
                    if sha256(pdf)!=doc['sha256']:
                        raise ValueError('PDF изменён после подготовки, запуск остановлен')
                    if multi_file.exists() and not single_file.exists():
                        raise ValueError('Найден Multi без Single — повреждённые результаты')
                    saved = json.loads(single_file.read_text(encoding='utf-8')) if single_file.exists() else None
                    if saved and saved.get('paper_sha256')!=doc['sha256']:
                        raise ValueError('Промежуточные результаты относятся к другому PDF')
                    def save_single(one):
                        one['repetition'] = repeat
                        one['paper_sha256'] = doc['sha256']
                        write_json(single_file, one)
                        logging.info('Single сохранён | проблем=%s | %.1f с | %s',len(one['problems']),one['seconds_to_single'],single_file)
                    solo, team = run_paper(doc, client, MODELS, OPTIONS, key, on_single_ready=save_single,saved_single=saved)
                    team['repetition'] = repeat
                    team['paper_sha256'] = doc['sha256']
                    write_json(multi_file, team)
                    logging.info('ГОТОВО | Single=%s проблем; Multi=%s проблем; раундов=%s; Single %.1f с; Multi %.1f с',len(solo['problems']),len(team['problems']),team['rounds'],solo['seconds_to_single'],team['seconds_end_to_end'])
                except Exception as exc:
                    logging.exception('ОШИБКА статьи %s: %s',key,exc)
                    issue = {'article':doc['filename'],'repeat':repeat,'error':repr(exc),'timestamp':datetime.now(timezone.utc).isoformat()}
                    failures.append(issue)
                    with (folder/'errors.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(issue,ensure_ascii=False)+'\n')
    finally:
        client.unload()
    logging.info('=== Обработка завершена | время %.1f с | ошибок %s | %s ===',time.perf_counter()-begin,len(failures),folder)
    # Семантическая оценка может потребовать отдельно загруженной embedding-модели.
    # Даже если она отсутствует, HTML должен быть сформирован по сохранённым JSON.
    try:
        from evaluate import auto_evaluate, read_json
        gold={x['paper_id']:x for x in read_json(HUMAN_GOLD)['papers']}
        if any((folder/m).exists() and any((folder/m).glob('*.json')) for m in ('single','multi')):
            auto_evaluate(folder,gold,threshold=SIMILARITY_THRESHOLD,
                          matcher=EVALUATION_MATCHER,embedding_model=EMBEDDING_MODEL)
    except Exception as exc:
        logging.warning('Предварительная семантическая оценка недоступна: %s. '
                        'Для оценки выполните ollama pull %s и python evaluate.py --name %s',
                        exc,EMBEDDING_MODEL,args.name)
    try:
        from visualize import report
        report(folder)
    except Exception:
        logging.exception('HTML не сформирован; попробуйте python visualize.py --name %s',args.name)
    if failures:
        raise SystemExit('Некоторые статьи завершились с ошибкой, проверьте experiment.log и повторите запуск.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Эксперимент с локальными моделями Ollama')
    parser.add_argument('--name', default='full', help='Название серии в output (full, pilot, и т.д.)')
    parser.add_argument('--limit', '--count', type=int,default=0,
                        help='Количество статей в партии (0=все оставшиеся). Например: --limit 5')
    parser.add_argument('--start',type=int,default=0,
                        help='Пропустить N статей в отсортированном списке (счёт с 0). Например --start 5 --limit 5')
    parser.add_argument('--paper',default='',help='Подстрока имени PDF, чтобы запустить одну статью')
    parser.add_argument('--repeats',type=int,default=REPEATS,help='Повторные прогоны каждой статьи')
    parser.add_argument('--check',action='store_true',help='Только проверить Ollama и модели')
    parser.add_argument('--resume-after-fix',action='store_true',help='Продолжить старую серию после исправления JSON с записью изменённых настроек; научно неоднородная серия')
    a = parser.parse_args()
    if a.repeats<1:
        parser.error('Число повторов должно быть не меньше 1')
    if a.limit < 0 or a.start < 0:
        parser.error('--start и --limit должны быть неотрицательными числами')
    run(a)
