"""Выполнение последовательных запросов к Ollama, выгрузка модели при её смене, опциональная телеметрия для графических процессоров NVIDIA."""
from __future__ import annotations
import json
import hashlib
import logging
import subprocess
import threading
import time
import re
from pathlib import Path
import psutil
import requests

class HardwareMonitor:
    def __init__(self, interval=0.5):
        self.interval=interval;self.values=[];self.stop=threading.Event();self.thread=None
    def snapshot(self):
        mem=psutil.virtual_memory();cpu=psutil.cpu_percent(interval=None)
        row={'timestamp':time.time(),'system_ram_used_bytes':mem.used,'system_ram_percent':mem.percent,'cpu_percent':cpu,
             'python_process_rss_bytes':psutil.Process().memory_info().rss}
        # Клиент Python и процесс Ollama, выполняющий генерацию, — это РАЗНЫЕ процессы.
        # Учитываем память обоих процессов; системная RAM также включает остальные процессы Windows.
        try:
            runners=[]
            for proc in psutil.process_iter(['name','memory_info']):
                name=(proc.info.get('name') or '').lower()
                if name.startswith('ollama'):
                    info=proc.info.get('memory_info')
                    if info:runners.append(info.rss)
            if runners:row['ollama_process_rss_bytes']=sum(runners)
        except (psutil.Error, OSError):pass
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
            gpu=r.stdout.strip().splitlines()[0].split(',')
            row.update({'gpu_mem_mb':float(gpu[0]),'gpu_util_percent':float(gpu[1]),
                        'gpu_temperature_c':float(gpu[2]),'gpu_power_w':float(gpu[3])})
        except (FileNotFoundError,subprocess.SubprocessError,IndexError,ValueError):pass
        return row
    def __enter__(self):
        self.values=[];self.stop.clear()
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start();return self
    def _loop(self):
        while not self.stop.is_set():
            self.values.append(self.snapshot());self.stop.wait(self.interval)
    def __exit__(self,*_):
        self.stop.set();self.thread.join(timeout=6)
    def summary(self):
        out={'samples':len(self.values)}
        for k in ['system_ram_used_bytes','system_ram_percent','cpu_percent','python_process_rss_bytes','ollama_process_rss_bytes','gpu_mem_mb','gpu_util_percent','gpu_temperature_c','gpu_power_w']:
            v=[x[k] for x in self.values if k in x]
            if v:out['max_'+k]=max(v);out['mean_'+k]=sum(v)/len(v)
        if len(self.values)>1 and all('gpu_power_w' in x for x in self.values):
            # Энергопотребление приблизительно рассчитано по периодическим замерам мощности GPU, а не измерено аппаратным счётчиком.
            out['estimated_gpu_joules']=sum((b['timestamp']-a['timestamp'])*a['gpu_power_w'] for a,b in zip(self.values,self.values[1:]))
        return out

def parse_json_object(answer):
    """Разобрать полный JSON-объект без автоматического добавления отсутствующих полей или скобок.

    Допускаются обычный JSON и JSON внутри блока Markdown. Если модель добавила
    пояснение перед ответом, ищем полный объект с первой открывающей фигурной скобки.
    При обрыве строки вызываем исключение, чтобы можно было повторить запрос к Ollama.
    """
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('Ollama returned an empty response')
    candidate = answer.strip()
    if candidate.startswith('```'):
        candidate = re.sub(r'^```(?:json)?\s*', '', candidate, count=1, flags=re.I)
        candidate = re.sub(r'\s*```$', '', candidate, count=1)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        pos = candidate.find('{')
        if pos < 0:
            raise
        parsed, index = json.JSONDecoder().raw_decode(candidate, pos)
        if candidate[index:].strip(' \t\r\n`'):
            raise ValueError('Unexpected text after JSON object')
    if not isinstance(parsed, dict):
        raise ValueError('Expected a JSON object at top level')
    return parsed


class OllamaClient:
    def __init__(self, url, cfg, telemetry_path):
        self.url=url.rstrip('/');self.cfg=cfg;self.telemetry_path=Path(telemetry_path)
        self.telemetry_path.parent.mkdir(parents=True,exist_ok=True)
        self.session=requests.Session();self.current=None
    def list_models(self):
        r=self.session.get(self.url+'/api/tags',timeout=30);r.raise_for_status()
        return {m['name']:m for m in r.json()['models']}
    def version(self):
        try:return self.session.get(self.url+'/api/version',timeout=5).json()
        except requests.RequestException:return {'unknown':True}
    def unload(self):
        if self.current:
            logging.info('  Выгрузка модели %s из памяти', self.current)
            try:self.session.post(self.url+'/api/generate',json={'model':self.current,'prompt':'','keep_alive':0,'stream':False},timeout=90).raise_for_status()
            except requests.RequestException:pass
            self.current=None
    def _save_invalid_response(self, request_id, attempt, response_text, server_response):
        """Сохранить исходный некорректный JSON для диагностики, не исправляя его автоматически."""
        folder=self.telemetry_path.parent/'invalid_json'
        folder.mkdir(parents=True,exist_ok=True)
        slug=re.sub(r'[^A-Za-z0-9_.-]+','_',request_id)[:70]
        suffix=hashlib.sha256(request_id.encode('utf-8')).hexdigest()[:10]
        path=folder/f'{slug}_{suffix}_attempt{attempt+1}.json'
        data={
            'request_id':request_id,
            'attempt':attempt+1,
            'raw_answer':response_text,
            'done_reason':server_response.get('done_reason'),
            'generated_tokens':server_response.get('eval_count'),
            'prompt_tokens':server_response.get('prompt_eval_count'),
        }
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        return str(path)

    def _save_server_error(self, request_id, attempt, status, response_body):
        """Записать сведения об ошибке Ollama на диск без сохранения полного промпта."""
        folder = self.telemetry_path.parent / 'server_errors'
        folder.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', request_id)[:70]
        suffix = hashlib.sha256(request_id.encode('utf-8')).hexdigest()[:10]
        path = folder / f'{slug}_{suffix}_attempt{attempt+1}.json'
        path.write_text(json.dumps({
            'request_id': request_id,
            'attempt': attempt + 1,
            'http_status': status,
            'response_body': response_body[:16000],
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        return str(path)

    def chat(self, model, role, prompt, *, request_id, system=None):
        if self.current != model:
            self.unload()
            logging.info('  Загрузка/выбор модели %s', model)
            self.current = model
        base_options = {k: self.cfg[k] for k in ('temperature', 'num_ctx', 'num_predict', 'seed') if k in self.cfg}
        base_options['num_ctx'] = int(self.cfg.get('role_num_ctx', {}).get(role, base_options.get('num_ctx',4096)))
        base_options['num_predict'] = int(self.cfg.get('role_num_predict', {}).get(role, base_options.get('num_predict',2048)))
        if role == 'referee' and request_id.endswith(':referee_repair'):
            base_options['num_predict'] = 750
        messages = [
            {'role': 'system', 'content': system or 'You are a rigorous research assistant.'},
            {'role': 'user', 'content': prompt},
        ]
        retries = max(0, int(self.cfg.get('retries', 3)))
        failures = []
        next_retry_reason = None
        logging.info('  Ollama вызов: %s | модель %s | id %s', role, model, request_id)

        for attempt in range(retries + 1):
            # Настройки повторных запросов записываются в telemetry,
            # а результаты с recovery отдельно помечаются как неоднородные.
            options = dict(base_options)
            this_messages = messages
            if next_retry_reason == 'invalid_json':
                options['num_predict'] = min(
                    max(int(base_options.get('num_predict', 2048)), int(self.cfg.get('retry_num_predict', 2600))),
                    max(750,int(options['num_ctx'])-1500))
                # Контекст уже назначен по роли; на сбой JSON увеличиваем только лимит ответа.
                # Для воспроизводимости seed не меняется при обычном повторе JSON.
                reminder = ('\nПовторный запрос после ошибки JSON. Верни ТОЛЬКО '
                            'завершённый JSON-объект, без Markdown. Пиши кратко.')
                this_messages = [
                    {'role': 'system', 'content': messages[0]['content'] + reminder},
                    messages[1],
                ]
            if next_retry_reason == 'repeat_loop':
                # Это НЕ ошибка переполнения окна. Модель вошла в цикл генерации.
                options['temperature'] = float(self.cfg.get('repeat_recovery_temperature', 0.2))
                options['repeat_penalty'] = float(self.cfg.get('repeat_recovery_penalty', 1.16))
                options['repeat_last_n'] = int(self.cfg.get('repeat_recovery_last_n', 128))
            if attempt:
                if self.current != model:
                    logging.info('  Повторная загрузка/выбор модели %s', model)
                    self.current = model
                logging.warning('  Повтор %d/%d | причина: %s | num_ctx=%s | num_predict=%s',
                                attempt, retries, next_retry_reason, options.get('num_ctx'),
                                options.get('num_predict'))

            payload = {
                'model': model, 'messages': this_messages, 'stream': False,
                'format': 'json', 'options': options,
                'keep_alive': self.cfg.get('keep_alive', '5m'),
                'think': self.cfg.get('think', False),
            }
            start = time.perf_counter()
            monitor = HardwareMonitor(self.cfg.get('sample_interval_seconds', 1))
            raw = {}
            failure_kind = None
            http_status = None
            server_error_file = None
            retryable = True
            try:
                with monitor:
                    res = self.session.post(
                        self.url + '/api/chat', json=payload,
                        timeout=self.cfg.get('timeout_seconds', 600),
                    )
                    res.raise_for_status()
                    raw = res.json()
                wall = time.perf_counter() - start
                answer = raw.get('message', {}).get('content', '')
                try:
                    parsed = parse_json_object(answer)
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    bad_path = self._save_invalid_response(request_id, attempt, answer, raw)
                    failure_kind = 'invalid_json'
                    raise ValueError(f'Некорректный JSON: {exc}; оригинал: {bad_path}') from exc
                perf = {
                    'request_id': request_id, 'role': role, 'model': model,
                    'attempt': attempt, 'success': True, 'retried': attempt > 0,
                    'num_predict': options.get('num_predict'), 'num_ctx': options.get('num_ctx'),
                    'temperature': options.get('temperature'), 'repeat_penalty': options.get('repeat_penalty'),
                    'repeat_last_n': options.get('repeat_last_n'),
                    'recovery_mode': next_retry_reason if attempt else None,
                    'seed': options.get('seed'), 'done_reason': raw.get('done_reason'),
                    'wall_seconds': wall, 'load_seconds': raw.get('load_duration', 0) / 1e9,
                    'prompt_eval_seconds': raw.get('prompt_eval_duration', 0) / 1e9,
                    'eval_seconds': raw.get('eval_duration', 0) / 1e9,
                    'total_server_seconds': raw.get('total_duration', 0) / 1e9,
                    'prompt_tokens': raw.get('prompt_eval_count', 0),
                    'generated_tokens': raw.get('eval_count', 0),
                    'tokens_per_second': round(raw.get('eval_count', 0) / (raw.get('eval_duration', 1) / 1e9), 2)
                        if raw.get('eval_duration') else None,
                    'prompt_tokens_per_second': round(raw.get('prompt_eval_count', 0) / (raw.get('prompt_eval_duration', 1) / 1e9), 2)
                        if raw.get('prompt_eval_duration') else None,
                    'hardware': monitor.summary(),
                }
                self._log(perf)
                logging.info('  Ответ %s: %.1f с | prompt=%s | ответ=%s токенов | %.2f ток/с | загрузка=%.1f с | VRAM peak=%s МБ',
                             role, wall, perf['prompt_tokens'], perf['generated_tokens'],
                             perf['tokens_per_second'] or 0, perf['load_seconds'],
                             perf['hardware'].get('max_gpu_mem_mb', 'н/д'))
                return parsed, perf
            except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
                wall = time.perf_counter() - start
                error_description = str(exc)
                if isinstance(exc, requests.HTTPError):
                    response = exc.response
                    http_status = getattr(response, 'status_code', None)
                    response_body = getattr(response, 'text', '') or ''
                    server_error_file = self._save_server_error(request_id, attempt, http_status, response_body)
                    compact = ' '.join(response_body.split())[:550]
                    error_description = f'HTTP {http_status}: {compact}; подробности: {server_error_file}'
                    failure_kind = 'http_error'
                    retryable = http_status in (429, 500, 502, 503, 504)
                    if http_status == 500:
                        if 'token repeat limit reached' in response_body.casefold():
                            failure_kind = 'repeat_loop'
                            logging.error('  Qwen/Ollama зациклилась при генерации токенов: это НЕ обязательно нехватка контекста. %s', compact)
                        else:
                            logging.error('  Ollama HTTP 500, причина из ответа сервера: %s', compact)
                elif isinstance(exc, requests.RequestException):
                    failure_kind = 'network_error'
                else:
                    failure_kind = failure_kind or 'invalid_json'
                failure = {
                    'request_id': request_id, 'role': role, 'model': model,
                    'attempt': attempt, 'success': False, 'wall_seconds': wall,
                    'failure_kind': failure_kind, 'http_status': http_status,
                    'server_error_file': server_error_file, 'error': error_description,
                    'num_predict': options.get('num_predict'), 'num_ctx': options.get('num_ctx'),
                    'temperature': options.get('temperature'), 'repeat_penalty': options.get('repeat_penalty'),
                    'repeat_last_n': options.get('repeat_last_n'),
                    'recovery_mode': next_retry_reason if attempt else None,
                    'seed': options.get('seed'), 'done_reason': raw.get('done_reason'),
                    'generated_tokens': raw.get('eval_count'),
                    'prompt_tokens': raw.get('prompt_eval_count'),
                    'hardware': monitor.summary(),
                }
                self._log(failure)
                failures.append(failure)
                logging.warning('  Сбой %s: попытка %d/%d: %s',
                                role, attempt + 1, retries + 1, error_description)
                if attempt >= retries or not retryable:
                    break
                if failure_kind == 'http_error' and http_status in (500, 502, 503, 504):
                    # Выгрузка модели после сбоя генерации может освободить выделенную память.
                    # НЕЛЬЗЯ незаметно уменьшать окно контекста или лимит ответа: другие
                    # параметры нарушат сопоставимость результатов контролируемого эксперимента.
                    self.unload()
                    next_retry_reason = 'repeat_loop' if failure_kind == 'repeat_loop' else f'http_{http_status}'
                elif failure_kind == 'invalid_json':
                    next_retry_reason = 'invalid_json'
                else:
                    next_retry_reason = failure_kind
                pause = min(12, 2 ** (attempt + 1))
                logging.warning('  Повтор запроса через %s сек. Результаты предыдущих статей сохранены.', pause)
                time.sleep(pause)
        raise RuntimeError(f'LLM {role}/{model}: {failures[-1]["error"]}')
    def _log(self,row):
        with self.telemetry_path.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
