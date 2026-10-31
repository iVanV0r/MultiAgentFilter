"""Настройки эксперимента. Изменения для новых серий сохраняются в итоговом JSON."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_PDFS = ROOT / 'data' / 'pdfs'
HUMAN_GOLD = ROOT / 'data' / 'gold.json'
OUTPUT = ROOT / 'output'
OLLAMA_URL = 'http://127.0.0.1:11434'

MODELS = {
    'proposer': 'qwen3.5:4b',
    'critic': 'phi4-mini:3.8b-q4_K_M',
    'referee': 'gemma4:e4b-it-q4_K_M',
}

OPTIONS = {
    'temperature': 0,
    'seed': 42,
    'think': False,
    'num_ctx': 4096,  # для Gemma 4 E4B экономим VRAM
    'role_num_ctx': {'proposer_chunk': 6144, 'proposer_merge': 6144, 'proposer_final': 6144,
                     'proposer_defense': 6144, 'critic': 6144, 'referee': 4096},
    'role_num_predict': {'proposer_chunk': 1100, 'proposer_merge': 1800, 'proposer_final': 1800,
                         'proposer_defense': 2200, 'critic': 1600, 'referee': 1700},
    'repeat_recovery_temperature': 0.2,
    'repeat_recovery_penalty': 1.16,
    'repeat_recovery_last_n': 128,
    'debate_context_chars': 3300,
    'global_context_chars': 3100,
    'judge_context_chars': 1800,
    'num_predict': 2048,
    'retry_num_predict': 2600,  # только при невалидном JSON; записывается в telemetry

    'chunk_chars': 5500,  # часть буфера окна оставляем для инструкции и ответа
    'overlap_chars': 350,
    'include_references': False,
    'max_rounds': 3,
    'stop_score': 8,
    'max_candidates_per_chunk': 3,
    'max_problems_per_paper': 7,
    'sample_interval_seconds': 1.0,
    'keep_alive': '5m',
    'reset_between_papers': True,
    'retries': 3,  # повторные попытки при HTTP 500/502/503/504, 429, сетевых ошибках и неверном JSON
    'timeout_seconds': 600,
}

REPEATS = 3
# Порог для семантического (bge-m3) сопоставления с человеческим эталоном.
# Он применяется ПОСЛЕ извлечения и не меняет решения Gemma.
# Порог эвристический: итоговые научные выводы требуют проверки человеком.
SIMILARITY_THRESHOLD = 0.70
# Для семантического сопоставления требуется отдельно установленная многоязычная модель эмбеддингов:
# ollama pull bge-m3
EMBEDDING_MODEL = 'bge-m3'
EVALUATION_MATCHER = 'semantic'

# Исключена из активного корпуса: повреждённый текст PDF / кодировка.
EXCLUDED_PAPER_IDS = {

}
